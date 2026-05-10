"""
TEE-Model — Sunum Oluşturucu
Playwright ile ekran görüntüsü alır, python-pptx ile sunum oluşturur.
"""

import asyncio
import subprocess
import time
import os
import sys
from pathlib import Path
from io import BytesIO

from playwright.async_api import async_playwright
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import pptx.oxml.ns as nsmap
from lxml import etree

# ---------------------------------------------------------------------------
# Renkler ve sabitler
# ---------------------------------------------------------------------------

BLUE_DARK  = RGBColor(0x1A, 0x23, 0x5E)   # koyu lacivert
BLUE_MID   = RGBColor(0x2E, 0x4A, 0xA8)   # orta mavi
BLUE_LIGHT = RGBColor(0xE8, 0xEF, 0xFA)   # açık mavi (arka plan)
ORANGE     = RGBColor(0xE8, 0x6A, 0x1A)   # vurgu turuncu
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_TEXT  = RGBColor(0x44, 0x44, 0x44)
GREEN      = RGBColor(0x27, 0xAE, 0x60)
RED        = RGBColor(0xC0, 0x39, 0x2B)
YELLOW     = RGBColor(0xF3, 0x9C, 0x12)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

APP_PORT = 8502   # Mock mod portu

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar — python-pptx
# ---------------------------------------------------------------------------

def add_rect(slide, x, y, w, h, fill_color=None, line_color=None, line_width=1):
    shape = slide.shapes.add_shape(
        pptx.enum.shapes.MSO_SHAPE_TYPE.AUTO_SHAPE,  # not used directly
        x, y, w, h
    )
    # We use add_shape with freeform or textbox; easier: use add_shape below
    return shape


def hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return RGBColor(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def add_colored_box(slide, x, y, w, h, fill_rgb, line_rgb=None, radius=False):
    """Dikdörtgen ekle."""
    from pptx.util import Pt
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    import pptx.oxml as oxml

    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if line_rgb:
        shape.line.color.rgb = line_rgb
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, text, x, y, w, h,
                 font_size=14, font_color=WHITE, bold=False,
                 align=PP_ALIGN.LEFT, wrap=True):
    txBox = slide.shapes.add_textbox(x, y, w, h)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.color.rgb = font_color
    run.font.bold = bold
    return txBox


def slide_background(slide, color_rgb):
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color_rgb


def add_header_bar(slide, title, subtitle=None):
    """Üst başlık çubuğu."""
    add_colored_box(slide, 0, 0, SLIDE_W, Inches(1.4), BLUE_DARK)
    add_text_box(slide, title,
                 Inches(0.4), Inches(0.1), Inches(12), Inches(0.8),
                 font_size=28, font_color=WHITE, bold=True)
    if subtitle:
        add_text_box(slide, subtitle,
                     Inches(0.4), Inches(0.85), Inches(12), Inches(0.45),
                     font_size=13, font_color=RGBColor(0xBB, 0xCC, 0xFF))


def add_footer(slide, text="TEE-Model | Maaş Mutemedi Onboarding Sistemi"):
    add_colored_box(slide, 0, Inches(7.15), SLIDE_W, Inches(0.35), BLUE_MID)
    add_text_box(slide, text,
                 Inches(0.3), Inches(7.17), Inches(12), Inches(0.3),
                 font_size=9, font_color=WHITE, align=PP_ALIGN.CENTER)


def add_screenshot(slide, img_path, x, y, w, h):
    if Path(img_path).exists():
        slide.shapes.add_picture(str(img_path), x, y, w, h)
    else:
        # Placeholder kutusu
        box = add_colored_box(slide, x, y, w, h, RGBColor(0xDD, 0xDD, 0xDD))
        add_text_box(slide, "[Ekran görüntüsü yok]",
                     x, y + h//2 - Inches(0.2), w, Inches(0.4),
                     font_size=11, font_color=GRAY_TEXT, align=PP_ALIGN.CENTER)


def add_arrow(slide, x1, y1, x2, y2, color=BLUE_MID):
    """İki nokta arasında ok çiz (connector)."""
    from pptx.util import Pt
    connector = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR_TYPE.STRAIGHT
        x1, y1, x2, y2
    )
    connector.line.color.rgb = color
    connector.line.width = Pt(2)


# ---------------------------------------------------------------------------
# Playwright: Ekran görüntüleri
# ---------------------------------------------------------------------------

async def take_screenshots():
    """Mock modda çalışan uygulamanın ekran görüntülerini al."""
    base_url = f"http://localhost:{APP_PORT}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1400, "height": 800}
        )
        page = await context.new_page()

        print("Uygulama yükleniyor...")
        try:
            await page.goto(base_url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=20000)
        except Exception as e:
            print(f"Sayfa yüklenemedi: {e}")
            await browser.close()
            return

        await asyncio.sleep(3)

        # 1. Ana sayfa — Tab 1 başlangıç durumu
        print("Screenshot 1: Ana sayfa")
        await page.screenshot(path=str(SCREENSHOTS_DIR / "01_ana_sayfa.png"), full_page=False)

        # 2. Tab 1 — Veritabanını Yenile butonuna tıkla
        print("Screenshot 2: Veri yükleme butonu tıklama")
        try:
            btn = page.get_by_role("button", name="Veritabanını Yenile")
            await btn.click()
            await asyncio.sleep(4)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "02_veri_yukleme_sonuc.png"))
        except Exception as e:
            print(f"  Buton bulunamadı: {e}")

        # 3. Tab 2 — Süreç Haritası
        print("Screenshot 3: Tab 2 - Süreç Haritası")
        try:
            tab2 = page.get_by_role("tab", name="Süreç Haritası ve Hata Kartları")
            await tab2.click()
            await asyncio.sleep(2)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "03_surec_haritasi_bos.png"))

            # Süreç Haritası Oluştur
            btn2 = page.get_by_role("button", name="Süreç Haritası Oluştur")
            await btn2.click()
            await asyncio.sleep(4)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "04_surec_haritasi_dolu.png"))
        except Exception as e:
            print(f"  Tab 2 hatası: {e}")

        # 4. Hata Kartları
        print("Screenshot 4: Hata Kartları")
        try:
            btn3 = page.get_by_role("button", name="Hata Kartlarını Oluştur")
            await btn3.click()
            await asyncio.sleep(4)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "05_hata_kartlari.png"))
        except Exception as e:
            print(f"  Hata kartı hatası: {e}")

        # 5. Tab 3 — Simülasyon
        print("Screenshot 5: Tab 3 - Simülasyon")
        try:
            tab3 = page.get_by_role("tab", name="İnteraktif Simülasyon")
            await tab3.click()
            await asyncio.sleep(2)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "06_simulasyon_bos.png"))

            btn4 = page.get_by_role("button", name="Yeni Senaryo Oluştur")
            await btn4.click()
            await asyncio.sleep(4)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "07_simulasyon_senaryo.png"))

            # Bir seçenek seç
            try:
                radio = page.locator("input[type='radio']").first
                await radio.click()
                await asyncio.sleep(1)
                confirm_btn = page.get_by_role("button", name="Cevabı Onayla")
                await confirm_btn.click()
                await asyncio.sleep(2)
                await page.screenshot(path=str(SCREENSHOTS_DIR / "08_simulasyon_cevap.png"))
            except Exception as e2:
                print(f"  Seçenek seçme hatası: {e2}")
        except Exception as e:
            print(f"  Tab 3 hatası: {e}")

        # 6. Tab 4 — Uzman Onay Paneli
        print("Screenshot 6: Tab 4 - Uzman Onay")
        try:
            tab4 = page.get_by_role("tab", name="Uzman Onay Paneli")
            await tab4.click()
            await asyncio.sleep(2)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "09_uzman_onay.png"))
        except Exception as e:
            print(f"  Tab 4 hatası: {e}")

        # 7. Tab 5 — Prompt Optimizer
        print("Screenshot 7: Tab 5 - Prompt Optimizer")
        try:
            tab5 = page.get_by_role("tab", name="Prompt Optimizer")
            await tab5.click()
            await asyncio.sleep(2)
            await page.screenshot(path=str(SCREENSHOTS_DIR / "10_prompt_optimizer.png"))
        except Exception as e:
            print(f"  Tab 5 hatası: {e}")

        await browser.close()
        print("Ekran görüntüleri tamamlandı.")


# ---------------------------------------------------------------------------
# PowerPoint Oluşturucu
# ---------------------------------------------------------------------------

def build_pptx():
    from pptx import Presentation
    from pptx.util import Inches, Pt, Emu
    from pptx.enum.text import PP_ALIGN
    import pptx.util

    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    blank_layout = prs.slide_layouts[6]  # Tamamen boş

    # =========================================================================
    # SLIDE 1 — Kapak
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, BLUE_DARK)

    # Üst dekoratif çubuk
    add_colored_box(slide, 0, 0, SLIDE_W, Inches(0.15), ORANGE)

    # Logo alanı (metin olarak)
    add_text_box(slide, "🏛️",
                 Inches(5.9), Inches(0.7), Inches(1.5), Inches(1.2),
                 font_size=60, align=PP_ALIGN.CENTER)

    # Ana başlık
    add_text_box(slide,
                 "TEE-Model",
                 Inches(1), Inches(1.8), Inches(11.33), Inches(1.2),
                 font_size=46, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    # Alt başlık
    add_text_box(slide,
                 "Tacit-Explicit Entegre Eğitim Modeli",
                 Inches(1), Inches(2.9), Inches(11.33), Inches(0.7),
                 font_size=24, font_color=ORANGE, bold=True, align=PP_ALIGN.CENTER)

    # Açıklama
    add_text_box(slide,
                 "Kamu kurumlarında kurumsal bellek kaybını azaltmak için\nyapay zeka destekli onboarding sistemi",
                 Inches(1.5), Inches(3.65), Inches(10.33), Inches(0.9),
                 font_size=16, font_color=RGBColor(0xBB, 0xCC, 0xFF), align=PP_ALIGN.CENTER)

    # Ayırıcı çizgi
    add_colored_box(slide, Inches(3), Inches(4.65), Inches(7.33), Inches(0.04), ORANGE)

    # Hedef kitle / tarih
    add_text_box(slide,
                 "Hedef Kullanım: Maaş Mutemedi Onboardingı  |  Nisan 2026",
                 Inches(1), Inches(4.8), Inches(11.33), Inches(0.4),
                 font_size=12, font_color=RGBColor(0x88, 0x99, 0xCC), align=PP_ALIGN.CENTER)

    # Alt çubuk
    add_colored_box(slide, 0, Inches(7.15), SLIDE_W, Inches(0.35), BLUE_MID)

    # =========================================================================
    # SLIDE 2 — Problem: Kurumsal Bellek Kaybı
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Problem: Kurumsal Bellek Kaybı",
                   "Kamu kurumlarında tecrübeli personel neden kritik önem taşır?")

    # Sol: problem açıklaması
    add_colored_box(slide, Inches(0.3), Inches(1.6), Inches(5.8), Inches(5.2), BLUE_LIGHT)

    problems = [
        ("📉", "Tecrübeli personel emekli olduğunda veya kurumdan ayrıldığında;"),
        ("🧠", "Yıllarca biriken örtük bilgi (tacit knowledge) kurumdan çıkar."),
        ("📋", "Resmi belgeler (mevzuat, yönergeler) yeterli rehberlik sağlamaz."),
        ("⏳", "Yeni personel aynı hataları tekrar yapar; öğrenme süreci uzar."),
        ("💰", "Maaş bordrosu gibi kritik süreçlerde hatalar, yasal ve mali risk doğurur."),
    ]
    y_pos = Inches(1.75)
    for icon, text in problems:
        add_text_box(slide, f"{icon}  {text}",
                     Inches(0.5), y_pos, Inches(5.4), Inches(0.7),
                     font_size=13, font_color=BLUE_DARK)
        y_pos += Inches(0.82)

    # Sağ: istatistikler / vurgu
    add_colored_box(slide, Inches(6.4), Inches(1.6), Inches(6.6), Inches(2.3), BLUE_DARK)
    add_text_box(slide, "Corporate Amnesia",
                 Inches(6.5), Inches(1.65), Inches(6.3), Inches(0.5),
                 font_size=18, font_color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide,
                 "Kurumsal bellek kaybı; deneyimli\n"
                 "çalışanların ayrılmasıyla oluşan\n"
                 "örtük bilgi boşluğudur.",
                 Inches(6.5), Inches(2.2), Inches(6.3), Inches(1.4),
                 font_size=13, font_color=WHITE, align=PP_ALIGN.CENTER)

    # Çözüm kutusu
    add_colored_box(slide, Inches(6.4), Inches(4.1), Inches(6.6), Inches(2.7), ORANGE)
    add_text_box(slide, "TEE-Model Çözümü",
                 Inches(6.5), Inches(4.15), Inches(6.3), Inches(0.5),
                 font_size=18, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide,
                 "✅ Tacit bilgiyi (uzman mülakatı) dijitalleştir\n"
                 "✅ Explicit bilgiyle (mevzuat) birleştir\n"
                 "✅ Yapay zeka ile eğitim materyali üret\n"
                 "✅ Yeni personele kişiselleştirilmiş rehberlik sun",
                 Inches(6.6), Inches(4.75), Inches(6.1), Inches(1.9),
                 font_size=12, font_color=WHITE)

    add_footer(slide)

    # =========================================================================
    # SLIDE 3 — Tacit vs Explicit Bilgi
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "İki Tür Bilgi: Tacit ve Explicit",
                   "TEE-Model her iki bilgi türünü bir araya getirir")

    # Tacit kutusu (sol)
    add_colored_box(slide, Inches(0.3), Inches(1.6), Inches(5.9), Inches(5.2), BLUE_DARK)
    add_text_box(slide, "🧠 Tacit (Örtük) Bilgi",
                 Inches(0.4), Inches(1.65), Inches(5.6), Inches(0.6),
                 font_size=20, font_color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    tacit_items = [
        "Deneyimle kazanılan, yazıya dökülmemiş bilgi",
        "\"Ocak ayında kümülatif matrahı unutma\"",
        "\"İcra yazısı gelince önce hukuku ara\"",
        "Uzman mülakatlarından elde edilir",
        "Kaynak: tacit_interview_raw.txt",
    ]
    y = Inches(2.35)
    for item in tacit_items:
        add_text_box(slide, f"• {item}",
                     Inches(0.5), y, Inches(5.4), Inches(0.55),
                     font_size=13, font_color=WHITE)
        y += Inches(0.6)

    # Explicit kutusu (sağ)
    add_colored_box(slide, Inches(7.1), Inches(1.6), Inches(5.9), Inches(5.2), BLUE_MID)
    add_text_box(slide, "📋 Explicit (Açık) Bilgi",
                 Inches(7.2), Inches(1.65), Inches(5.6), Inches(0.6),
                 font_size=20, font_color=ORANGE, bold=True, align=PP_ALIGN.CENTER)
    explicit_items = [
        "Resmi belgelerde yazılı, standartlaşmış bilgi",
        "Mevzuat, yönetmelik, genelge",
        "Form ve prosedür açıklamaları",
        "Her zaman erişilebilir, güncellenebilir",
        "Kaynak: explicit_mevzuat.txt",
    ]
    y = Inches(2.35)
    for item in explicit_items:
        add_text_box(slide, f"• {item}",
                     Inches(7.3), y, Inches(5.4), Inches(0.55),
                     font_size=13, font_color=WHITE)
        y += Inches(0.6)

    # Ortada + işareti
    add_colored_box(slide, Inches(6.0), Inches(3.6), Inches(1.1), Inches(1.1),
                    ORANGE, line_rgb=WHITE)
    add_text_box(slide, "+",
                 Inches(6.0), Inches(3.65), Inches(1.1), Inches(0.9),
                 font_size=40, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    # Alt: birleşme
    add_colored_box(slide, Inches(3.5), Inches(6.85), Inches(6.3), Inches(0.45), ORANGE)
    add_text_box(slide, "→  Birlikte: Kapsamlı, Güvenilir Eğitim İçeriği",
                 Inches(3.5), Inches(6.88), Inches(6.3), Inches(0.4),
                 font_size=13, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 4 — Sistem Mimarisi Genel Bakış
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sistem Mimarisi",
                   "Veriden yapay zeka destekli eğitim içeriğine uzanan akış")

    # Mimari bileşenler — yatay akış
    components = [
        ("📄\nVeri\nKaynakları", BLUE_DARK,   Inches(0.2)),
        ("🔒\nAnonimleştirme\n(KVKK)", BLUE_MID,    Inches(2.1)),
        ("🧩\nChunking &\nEmbedding", RGBColor(0x16, 0x79, 0xA1), Inches(4.0)),
        ("🗄️\nVektör\nVeritabanı", RGBColor(0x1E, 0x8B, 0x4C), Inches(5.9)),
        ("🔍\nSemantik\nArama", RGBColor(0x7D, 0x3C, 0x98), Inches(7.8)),
        ("🤖\nRAG +\nGemini", ORANGE,          Inches(9.7)),
        ("📊\nStreamlit\nArayüzü", BLUE_DARK,   Inches(11.6)),
    ]

    box_w = Inches(1.7)
    box_h = Inches(1.5)
    box_y = Inches(2.5)
    arrow_y = box_y + box_h / 2

    for label, color, x in components:
        add_colored_box(slide, x, box_y, box_w, box_h, color)
        add_text_box(slide, label, x, box_y + Inches(0.05), box_w, box_h - Inches(0.1),
                     font_size=11, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    # Oklar
    for i in range(len(components) - 1):
        _, _, x_cur = components[i]
        _, _, x_next = components[i+1]
        add_arrow(slide,
                  x_cur + box_w, arrow_y,
                  x_next, arrow_y,
                  GRAY_TEXT)

    # Alt açıklamalar
    descriptions = [
        ("Mülakatlar\n+ Mevzuat", Inches(0.2)),
        ("PII Maskeleme\n(regex)", Inches(2.1)),
        ("gemini-\nembedding-001", Inches(4.0)),
        ("ChromaDB\n(yerel)", Inches(5.9)),
        ("Cosine\nBenzerlik", Inches(7.8)),
        ("gemini-\n2.5-flash", Inches(9.7)),
        ("4 sekme\nUI", Inches(11.6)),
    ]
    for desc, x in descriptions:
        add_text_box(slide, desc, x, box_y + box_h + Inches(0.1), box_w, Inches(0.7),
                     font_size=10, font_color=GRAY_TEXT, align=PP_ALIGN.CENTER)

    # Grounding notu
    add_colored_box(slide, Inches(0.3), Inches(5.1), Inches(12.7), Inches(1.0),
                    BLUE_LIGHT, line_rgb=BLUE_MID)
    add_text_box(slide,
                 "🔗  Grounding (Dayandırma): Model YALNIZCA veritabanından getirilen chunk'lara dayanır. "
                 "Halüsinasyon önleme için bağlamda olmayan hiçbir bilgi üretilmez.",
                 Inches(0.5), Inches(5.15), Inches(12.3), Inches(0.8),
                 font_size=13, font_color=BLUE_DARK)

    # Veri kaynakları detayı
    add_colored_box(slide, Inches(0.3), Inches(6.2), Inches(12.7), Inches(0.8), BLUE_LIGHT)
    add_text_box(slide,
                 "Veri Kaynakları:  📄 explicit_mevzuat.txt  (resmi mevzuat, formlar, prosedürler)   |   "
                 "🎙️ tacit_interview_raw.txt  (deneyimli personel mülakatı — KVKK kapsamında anonimleştirilir)",
                 Inches(0.5), Inches(6.25), Inches(12.3), Inches(0.6),
                 font_size=11, font_color=GRAY_TEXT)

    add_footer(slide)

    # =========================================================================
    # SLIDE 5 — RAG Akış Diyagramı (Prompt Nasıl Oluşturulur)
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "RAG Akışı: Prompt Nasıl Oluşturulur?",
                   "Kullanıcı isteğinden yanıta kadar adım adım")

    # ---- Adımlar dikey akış ----
    steps = [
        ("1", "Kullanıcı İsteği",
         "Kullanıcı bir sekmeye tıklar\n(Süreç Haritası, Hata Kartı, Simülasyon…)",
         BLUE_DARK, Inches(0.3)),

        ("2", "Sorgu Belirleme",
         "Her generator için önceden tanımlı\n(veya optimize edilmiş) Türkçe arama sorgusu seçilir",
         BLUE_MID, Inches(0.3)),

        ("3", "Embedding & Arama",
         "Sorgu, gemini-embedding-001 ile vektöre dönüştürülür.\nChromaDB'de cosine benzerliğiyle en yakın chunk'lar bulunur",
         RGBColor(0x16, 0x79, 0xA1), Inches(0.3)),

        ("4", "Bağlam Metni Oluşturma",
         "Seçilen chunk'lar birleştirilerek BAĞLAM bloğu oluşturulur.\nChunk indeksleri kaynak atıfı için saklanır",
         RGBColor(0x1E, 0x8B, 0x4C), Inches(0.3)),

        ("5", "Prompt Birleştirme",
         "Sistem talimatı + BAĞLAM + görev talimatı birleştirilerek\ntam prompt oluşturulur",
         RGBColor(0x7D, 0x3C, 0x98), Inches(0.3)),

        ("6", "LLM Çağrısı",
         "Prompt gemini-2.5-flash'a gönderilir.\nModel YALNIZCA bağlamdan JSON formatında yanıt üretir",
         ORANGE, Inches(0.3)),

        ("7", "JSON Doğrulama & Gösterim",
         "Yanıt JSON olarak ayrıştırılır (hata varsa yeniden denenir).\nStreamlit arayüzünde tablolar ve kartlar halinde gösterilir",
         BLUE_DARK, Inches(0.3)),
    ]

    box_h2 = Inches(0.72)
    box_w2 = Inches(4.2)
    box_w_desc = Inches(7.8)
    start_y = Inches(1.55)
    gap = Inches(0.08)
    num_w = Inches(0.5)

    for i, (num, title, desc, color, _) in enumerate(steps):
        y = start_y + i * (box_h2 + gap)
        # Numara dairesi
        add_colored_box(slide, Inches(0.2), y, num_w, box_h2, color)
        add_text_box(slide, num, Inches(0.2), y, num_w, box_h2,
                     font_size=18, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        # Başlık kutusu
        add_colored_box(slide, Inches(0.75), y, box_w2, box_h2, color)
        add_text_box(slide, title, Inches(0.85), y + Inches(0.15), box_w2 - Inches(0.2), box_h2,
                     font_size=14, font_color=WHITE, bold=True)
        # Açıklama kutusu
        add_colored_box(slide, Inches(5.1), y, box_w_desc, box_h2, BLUE_LIGHT)
        add_text_box(slide, desc, Inches(5.2), y + Inches(0.05), box_w_desc - Inches(0.2), box_h2,
                     font_size=11, font_color=BLUE_DARK)

    # Aşağı oklar (sol taraf)
    for i in range(len(steps) - 1):
        y_cur = start_y + i * (box_h2 + gap) + box_h2
        y_nxt = start_y + (i+1) * (box_h2 + gap)
        add_colored_box(slide,
                        Inches(0.43), y_cur,
                        Inches(0.04), y_nxt - y_cur,
                        GRAY_TEXT)

    add_footer(slide)

    # =========================================================================
    # SLIDE 6 — Prompt Yapısı (Detay)
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Prompt Yapısı: Grounding Şablonu",
                   "Model yalnızca belgelerden gelen bilgiyi kullanır")

    # Sol: prompt bileşenleri
    add_colored_box(slide, Inches(0.3), Inches(1.55), Inches(5.5), Inches(5.6), BLUE_LIGHT)
    add_text_box(slide, "Prompt Bileşenleri",
                 Inches(0.4), Inches(1.6), Inches(5.2), Inches(0.5),
                 font_size=16, font_color=BLUE_DARK, bold=True, align=PP_ALIGN.CENTER)

    parts = [
        (BLUE_DARK, "① Sistem Talimatı",
         "\"Sen bir kamu kurumu eğitim içerik uzmanısın.\nYALNIZCA sağlanan bağlam belgelerini kullan.\""),
        (RGBColor(0x1E, 0x8B, 0x4C), "② Bağlam Bloğu",
         "Veritabanından getirilen chunk'lar:\n[Kaynak 1: mevzuat.txt — Chunk 3]\n... metin içeriği ..."),
        (RGBColor(0x7D, 0x3C, 0x98), "③ Chunk İndeksleri",
         "KAYNAK CHUNK İNDEKSLERİ: [0, 3, 7, 12]"),
        (ORANGE, "④ Görev Talimatı",
         "\"Aşağıdaki JSON şemasını kullan ve\nBAŞKA HİÇBİR ŞEY yazma: { ... }\""),
    ]

    y = Inches(2.2)
    for color, title, content in parts:
        add_colored_box(slide, Inches(0.4), y, Inches(5.2), Inches(0.35), color)
        add_text_box(slide, title, Inches(0.5), y + Inches(0.04), Inches(5.0), Inches(0.28),
                     font_size=11, font_color=WHITE, bold=True)
        add_text_box(slide, content, Inches(0.45), y + Inches(0.38), Inches(5.1), Inches(0.65),
                     font_size=10, font_color=GRAY_TEXT)
        y += Inches(1.1)

    # Sağ: sonuç akışı
    add_colored_box(slide, Inches(6.1), Inches(1.55), Inches(6.9), Inches(5.6), BLUE_DARK)
    add_text_box(slide, "Yanıt Akışı",
                 Inches(6.2), Inches(1.6), Inches(6.6), Inches(0.5),
                 font_size=16, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    flow_items = [
        (ORANGE,     "gemini-2.5-flash",     "Prompt alır, sadece bağlamdaki\nbilgilerle JSON üretir"),
        (RGBColor(0x1E, 0x8B, 0x4C), "JSON Doğrulama", "Markdown temizlenir, parse edilir.\nHata → otomatik tekrar denenir"),
        (BLUE_MID,   "Streamlit Gösterimi",  "Tablolar, hata kartları,\nsimülasyon soruları olarak sunulur"),
        (RGBColor(0x7D, 0x3C, 0x98), "Kaynak Atıfı",   "Her içerik parçasında chunk\nindeksleri gösterilir → şeffaf izleme"),
    ]

    y2 = Inches(2.2)
    for color, title, desc in flow_items:
        add_colored_box(slide, Inches(6.2), y2, Inches(6.6), Inches(1.0), color)
        add_text_box(slide, title, Inches(6.3), y2 + Inches(0.05), Inches(6.3), Inches(0.35),
                     font_size=13, font_color=WHITE, bold=True)
        add_text_box(slide, desc, Inches(6.3), y2 + Inches(0.4), Inches(6.3), Inches(0.55),
                     font_size=11, font_color=RGBColor(0xCC, 0xDD, 0xFF))
        if y2 + Inches(1.0) < Inches(7.0):
            add_colored_box(slide, Inches(9.3), y2 + Inches(1.0), Inches(0.06), Inches(0.25), ORANGE)
        y2 += Inches(1.25)

    add_footer(slide)

    # =========================================================================
    # SLIDE 7 — Teknoloji Yığını
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Teknoloji Yığını",
                   "Kurumun kendi altyapısında çalışabilen, açık kaynaklı bileşenler")

    tech_items = [
        ("🤖", "Gemini 2.5 Flash", "İçerik üretimi (RAG)\nGoogle'ın en hızlı LLM'i",
         BLUE_DARK, Inches(0.3), Inches(1.6)),
        ("📐", "Gemini Embedding 001", "Metni vektöre dönüştürme\nAnlamsal arama için",
         BLUE_MID, Inches(3.55), Inches(1.6)),
        ("🗄️", "ChromaDB", "Yerel vektör veritabanı\nKurumda kalır, buluta gitmez",
         RGBColor(0x16, 0x79, 0xA1), Inches(6.8), Inches(1.6)),
        ("🌐", "Streamlit", "Python tabanlı web arayüzü\nHızlı prototipleme",
         RGBColor(0x7D, 0x3C, 0x98), Inches(10.05), Inches(1.6)),
        ("🔒", "Regex Anonimleştirici", "KVKK uyumlu PII maskeleme\nİsim, TC, IBAN, telefon",
         ORANGE, Inches(0.3), Inches(4.1)),
        ("🐍", "Python 3.12+", "Ana programlama dili\nTüm bileşenler Python'da",
         RGBColor(0x1E, 0x8B, 0x4C), Inches(3.55), Inches(4.1)),
        ("📊", "Pandas", "Veri tabloları ve raporlama\nArayüz içi veri gösterimi",
         GRAY_TEXT, Inches(6.8), Inches(4.1)),
        ("🔬", "Prompt Optimizer", "3 varyant dene → LLM hakemi\nEn iyi prompt'u kaydet",
         BLUE_DARK, Inches(10.05), Inches(4.1)),
    ]

    for icon, name, desc, color, x, y in tech_items:
        add_colored_box(slide, x, y, Inches(3.0), Inches(2.2), color)
        add_text_box(slide, icon, x, y + Inches(0.1), Inches(3.0), Inches(0.7),
                     font_size=30, align=PP_ALIGN.CENTER)
        add_text_box(slide, name, x, y + Inches(0.75), Inches(3.0), Inches(0.5),
                     font_size=14, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)
        add_text_box(slide, desc, x, y + Inches(1.25), Inches(3.0), Inches(0.85),
                     font_size=11, font_color=RGBColor(0xCC, 0xDD, 0xFF), align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 8 — Ekran Görüntüsü: Ana Sayfa
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Arayüz: Ana Sayfa ve Sekmeler",
                   "Streamlit tabanlı 5 sekmeli web arayüzü")

    add_screenshot(slide,
                   SCREENSHOTS_DIR / "01_ana_sayfa.png",
                   Inches(0.3), Inches(1.55), Inches(12.7), Inches(5.3))

    add_colored_box(slide, Inches(0.3), Inches(6.9), Inches(12.7), Inches(0.35), BLUE_LIGHT)
    add_text_box(slide,
                 "5 sekme: Veri Yükleme  |  Süreç Haritası & Hata Kartları  |  Simülasyon  |  Uzman Onay  |  Prompt Optimizer",
                 Inches(0.4), Inches(6.92), Inches(12.5), Inches(0.3),
                 font_size=11, font_color=BLUE_DARK, align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 9 — Ekran Görüntüsü: Veri Yükleme
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sekme 1: Veri Yükleme ve İşleme",
                   "\"Veritabanını Yenile\" butonuna tıklandıktan sonra")

    # İki screenshot yan yana
    add_screenshot(slide,
                   SCREENSHOTS_DIR / "01_ana_sayfa.png",
                   Inches(0.3), Inches(1.6), Inches(6.1), Inches(4.0))
    add_screenshot(slide,
                   SCREENSHOTS_DIR / "02_veri_yukleme_sonuc.png",
                   Inches(6.6), Inches(1.6), Inches(6.4), Inches(4.0))

    # Açıklama
    add_text_box(slide, "Önce (Buton tıklanmadı)",
                 Inches(0.3), Inches(5.65), Inches(6.1), Inches(0.4),
                 font_size=12, font_color=BLUE_MID, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Sonra (Veri yüklendi)",
                 Inches(6.6), Inches(5.65), Inches(6.4), Inches(0.4),
                 font_size=12, font_color=GREEN, bold=True, align=PP_ALIGN.CENTER)

    add_colored_box(slide, Inches(0.3), Inches(6.1), Inches(12.7), Inches(0.65), BLUE_LIGHT)
    add_text_box(slide,
                 "Süreç: Buton tıklanır → Anonymizer PII maskeler → Chunking yapılır → "
                 "Embedding hesaplanır → ChromaDB'ye kaydedilir → Özet istatistikler gösterilir",
                 Inches(0.5), Inches(6.13), Inches(12.3), Inches(0.55),
                 font_size=11, font_color=BLUE_DARK)

    add_footer(slide)

    # =========================================================================
    # SLIDE 10 — Ekran Görüntüsü: Süreç Haritası
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sekme 2: Süreç Haritası",
                   "\"Süreç Haritası Oluştur\" butonuna basıldıktan sonra AI içerik üretir")

    add_screenshot(slide,
                   SCREENSHOTS_DIR / "04_surec_haritasi_dolu.png",
                   Inches(0.3), Inches(1.55), Inches(12.7), Inches(5.3))

    add_colored_box(slide, Inches(0.3), Inches(6.9), Inches(12.7), Inches(0.35), BLUE_LIGHT)
    add_text_box(slide,
                 "RAG ile üretilen adımlar: Kadro güncelleme → Göreve başlama belgesi → Ek ödeme/kesinti → Amir onayı → Muhasebe teslimi",
                 Inches(0.5), Inches(6.92), Inches(12.3), Inches(0.3),
                 font_size=11, font_color=BLUE_DARK, align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 11 — Ekran Görüntüsü: Hata Kartları
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sekme 2: Hata Kartları",
                   "Yeni mutemedlerin en sık yaptığı hatalar — tacit bilgiden türetilir")

    add_screenshot(slide,
                   SCREENSHOTS_DIR / "05_hata_kartlari.png",
                   Inches(0.3), Inches(1.55), Inches(12.7), Inches(5.3))

    add_colored_box(slide, Inches(0.3), Inches(6.9), Inches(12.7), Inches(0.35), BLUE_LIGHT)
    add_text_box(slide,
                 "Her kart: Hata tanımı | Kök neden | Tespit yöntemi | Doğru uygulama | Kaynak chunk indeksleri",
                 Inches(0.5), Inches(6.92), Inches(12.3), Inches(0.3),
                 font_size=11, font_color=BLUE_DARK, align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 12 — Ekran Görüntüsü: Simülasyon
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sekme 3: İnteraktif Simülasyon",
                   "Senaryo bazlı pratik sorular — doğru/yanlış geri bildirim")

    add_screenshot(slide,
                   SCREENSHOTS_DIR / "07_simulasyon_senaryo.png",
                   Inches(0.3), Inches(1.55), Inches(6.1), Inches(5.3))
    add_screenshot(slide,
                   SCREENSHOTS_DIR / "08_simulasyon_cevap.png",
                   Inches(6.6), Inches(1.55), Inches(6.4), Inches(5.3))

    add_text_box(slide, "Senaryo görünümü",
                 Inches(0.3), Inches(6.9), Inches(6.1), Inches(0.35),
                 font_size=11, font_color=BLUE_MID, bold=True, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Cevap seçildikten sonra anlık geri bildirim",
                 Inches(6.6), Inches(6.9), Inches(6.4), Inches(0.35),
                 font_size=11, font_color=GREEN, bold=True, align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 13 — Ekran Görüntüsü: Uzman Onay Paneli
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sekme 4: Uzman Onay Paneli",
                   "AI içerikleri eğitime aktarılmadan önce uzman incelemesinden geçer")

    add_screenshot(slide,
                   SCREENSHOTS_DIR / "09_uzman_onay.png",
                   Inches(0.3), Inches(1.55), Inches(12.7), Inches(5.1))

    # Açıklama kutusu
    add_colored_box(slide, Inches(0.3), Inches(6.7), Inches(12.7), Inches(0.55), BLUE_LIGHT)
    add_text_box(slide,
                 "Her içerik için: ✓ Onayla  |  ✗ Reddet  |  Onay özeti (Toplam / Onaylanan / Reddedilen / Bekleyen)  "
                 "|  Onaylanmayanlar eğitime aktarılmaz",
                 Inches(0.5), Inches(6.73), Inches(12.3), Inches(0.45),
                 font_size=11, font_color=BLUE_DARK, align=PP_ALIGN.CENTER)

    add_footer(slide)

    # =========================================================================
    # SLIDE 14 — Prompt Optimizer
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Sekme 5: Prompt Optimizer",
                   "En iyi prompt'u otomatik bul — LLM hakemli değerlendirme")

    # Sol: açıklama
    add_colored_box(slide, Inches(0.3), Inches(1.6), Inches(6.0), Inches(5.2), BLUE_DARK)
    add_text_box(slide, "Nasıl Çalışır?",
                 Inches(0.4), Inches(1.65), Inches(5.7), Inches(0.5),
                 font_size=16, font_color=ORANGE, bold=True, align=PP_ALIGN.CENTER)

    opt_steps = [
        "① 3 farklı sorgu + talimat varyantı tanımlanır",
        "② Her varyant için içerik üretilir",
        "③ Gemini hakem olarak her varyantı puanlar:",
        "    • Dayandırma (grounding): 1-10",
        "    • Tamlama (completeness): 1-10",
        "    • Pratik Fayda (usefulness): 1-10",
        "④ En yüksek toplam puana sahip prompt kaydedilir",
        "⑤ Sonraki içerik üretiminde otomatik kullanılır",
    ]
    y = Inches(2.3)
    for step in opt_steps:
        add_text_box(slide, step, Inches(0.5), y, Inches(5.5), Inches(0.45),
                     font_size=12, font_color=WHITE)
        y += Inches(0.5)

    # Sağ: screenshot
    add_screenshot(slide,
                   SCREENSHOTS_DIR / "10_prompt_optimizer.png",
                   Inches(6.5), Inches(1.6), Inches(6.5), Inches(5.2))

    add_footer(slide)

    # =========================================================================
    # SLIDE 15 — KVKK & Güvenlik
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, WHITE)
    add_header_bar(slide, "Veri Gizliliği ve Güvenlik",
                   "KVKK uyumlu tasarım — hassas veriler korunur")

    items = [
        (BLUE_DARK,  "🔒 PII Anonimleştirme",
         "Kişisel veriler (isim, TC kimlik, IBAN, telefon) regex ile otomatik maskelenir. "
         "Maskeleme işlemi veri kaynaktan çıkmadan gerçekleşir."),
        (BLUE_MID,   "🏠 Yerel Vektör Veritabanı",
         "ChromaDB tamamen yerel çalışır. Döküman içerikleri buluta gönderilmez. "
         "Kurumun kendi sunucusunda barındırılabilir."),
        (GREEN,      "🔍 Şeffaf Kaynak Atıfı",
         "Her üretilen içerik, hangi kaynak chunk'lardan türetildiğini gösterir. "
         "İçerik kalitesi izlenebilir ve denetlenebilir."),
        (ORANGE,     "✅ İnsan Onay Döngüsü",
         "Hiçbir AI içeriği otomatik olarak eğitim materyaline dönüşmez. "
         "Uzman onay panelinden geçmeden içerik kullanıma alınmaz."),
        (RGBColor(0x7D, 0x3C, 0x98), "📝 Denetim Logu",
         "Anonimleştirme kaydı (mask_log) arayüzde gösterilir ve saklanabilir. "
         "Hangi verinin nasıl maskelendiği her zaman incelenebilir."),
    ]

    y = Inches(1.6)
    for color, title, desc in items:
        add_colored_box(slide, Inches(0.3), y, Inches(12.7), Inches(0.9), color)
        add_text_box(slide, title, Inches(0.5), y + Inches(0.05), Inches(3.5), Inches(0.8),
                     font_size=14, font_color=WHITE, bold=True)
        add_text_box(slide, desc, Inches(4.0), y + Inches(0.08), Inches(9.0), Inches(0.75),
                     font_size=12, font_color=WHITE)
        y += Inches(1.0)

    add_footer(slide)

    # =========================================================================
    # SLIDE 16 — Sonuç ve Katkılar
    # =========================================================================
    slide = prs.slides.add_slide(blank_layout)
    slide_background(slide, BLUE_DARK)

    add_colored_box(slide, 0, 0, SLIDE_W, Inches(0.15), ORANGE)

    add_text_box(slide, "Sonuç ve Katkılar",
                 Inches(0.5), Inches(0.5), Inches(12.3), Inches(0.8),
                 font_size=32, font_color=WHITE, bold=True, align=PP_ALIGN.CENTER)

    add_colored_box(slide, Inches(4.5), Inches(1.25), Inches(4.3), Inches(0.04), ORANGE)

    contributions = [
        ("🎯", "Akademik Katkı",
         "Tacit ve explicit bilginin RAG mimarisinde\nbütünleştirilmesi için yenilikiçi bir çerçeve"),
        ("🏛️", "Pratik Katkı",
         "Kamu kurumlarında onboarding süresini\nkısaltan, denetlenebilir bir production sistemi"),
        ("🔒", "Etik Katkı",
         "KVKK uyumlu, şeffaf kaynak atıflı ve\ninsan onayı gerektiren sorumlu AI kullanımı"),
        ("🔬", "Teknik Katkı",
         "Prompt Optimizer ile otomatik prompt\niyileştirme döngüsü (autoresearch ilhamı)"),
    ]

    for i, (icon, title, desc) in enumerate(contributions):
        x = Inches(0.3) + (i % 2) * Inches(6.5)
        y = Inches(1.6) + (i // 2) * Inches(2.6)
        add_colored_box(slide, x, y, Inches(6.0), Inches(2.4), BLUE_MID)
        add_text_box(slide, f"{icon}  {title}",
                     x + Inches(0.1), y + Inches(0.1), Inches(5.7), Inches(0.55),
                     font_size=16, font_color=ORANGE, bold=True)
        add_text_box(slide, desc,
                     x + Inches(0.2), y + Inches(0.7), Inches(5.5), Inches(1.5),
                     font_size=13, font_color=WHITE)

    add_colored_box(slide, 0, Inches(7.15), SLIDE_W, Inches(0.35), ORANGE)
    add_text_box(slide, "TEE-Model | Nisan 2026",
                 Inches(0.3), Inches(7.17), SLIDE_W - Inches(0.6), Inches(0.3),
                 font_size=10, font_color=WHITE, align=PP_ALIGN.CENTER)

    # =========================================================================
    # Kaydet
    # =========================================================================
    output_path = Path(__file__).parent / "TEE_Model_Sunum.pptx"
    prs.save(str(output_path))
    print(f"\n✅ Sunum kaydedildi: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------

async def main():
    # 1. Mock modda uygulama başlat
    print(f"Mock modda uygulama başlatılıyor (port {APP_PORT})...")
    project_dir = Path(__file__).parent
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(APP_PORT),
         "--server.headless", "true",
         "--server.runOnSave", "false"],
        env={**os.environ, "MOCK_MODE": "true"},
        cwd=str(project_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Uygulamanın başlamasını bekle
    print("Uygulama başlaması bekleniyor...")
    time.sleep(8)

    try:
        # 2. Ekran görüntüsü al
        await take_screenshots()
    finally:
        # 3. Uygulamayı durdur
        proc.terminate()
        proc.wait()
        print("Mock uygulama durduruldu.")

    # 4. Sunumu oluştur
    print("\nPowerPoint oluşturuluyor...")
    output = build_pptx()
    return output


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nTamamlandı: {result}")
