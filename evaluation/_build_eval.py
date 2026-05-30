"""One-shot builder for delf_questions.json v2.0.

Preserves the original 18 questions (adds expected_sources remapped to REAL corpus
files + new fields), then appends 32 NEW bilingual questions grounded in chunk text
actually read from ChromaDB. All ground truths flagged UNVALIDATED_NEEDS_DELF_EXPERT.

Run once:  python evaluation/_build_eval.py
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
backup = json.loads((BASE / "delf_questions.v1_backup.json").read_text(encoding="utf-8"))
old = backup["questions"]
old_by_id = {q["id"]: q for q in old}

# --- Remap table for the original 18 (verified against chunk content) ----------
# language: which language the ORIGINAL `question` field is in.
# All filenames below VERIFIED present in evaluation/corpus_files.json (real source_filename
# metadata in ChromaDB) and content-checked against sampled chunk text.
REMAP = {
    "d01": {"sources": ["manuel-exacor.pdf", "Diaporama_rôle_EC_VF.pptx"], "lang": "tr", "type": "factoid", "difficulty": "easy"},
    "d02": {"sources": ["manuel-exacor.pdf"], "lang": "tr", "type": "procedural", "difficulty": "hard"},
    "d03": {"sources": ["manuel-exacor.pdf"], "lang": "tr", "type": "procedural", "difficulty": "medium"},
    "d04": {"sources": ["manuel-exacor.pdf", "Introduction_critères_corriges.docx"], "lang": "tr", "type": "factoid", "difficulty": "medium"},
    "d05": {"sources": ["manuel-exacor.pdf", "Présentation_copies_atypiques_VF.pptx"], "lang": "tr", "type": "analytical", "difficulty": "hard"},
    "d06": {"sources": ["B2_Grille_PO.pdf", "PO_B2.pdf"], "lang": "tr", "type": "factoid", "difficulty": "easy"},
    "d07": {"sources": ["B2_Grille_PO.pdf", "PO_B2.pdf"], "lang": "tr", "type": "factoid", "difficulty": "medium"},
    "d08": {"sources": ["A1_Descripteurs_PO.pdf", "A1_Grille_PO.pdf"], "lang": "tr", "type": "definition", "difficulty": "medium"},
    "d09": {"sources": ["Echelle globale.pdf"], "lang": "tr", "type": "definition", "difficulty": "easy"},
    "d10": {"sources": ["manuel-exacor.pdf", "Diaporama_rôle_EC_VF.pptx"], "lang": "fr", "type": "factoid", "difficulty": "easy"},
    "d11": {"sources": ["manuel-exacor.pdf"], "lang": "fr", "type": "procedural", "difficulty": "hard"},
    "d12": {"sources": ["PO_B1.pdf", "B1_Grille_PO.pdf"], "lang": "fr", "type": "analytical", "difficulty": "medium"},
    "d13": {"sources": ["manuel-exacor.pdf"], "lang": "fr", "type": "procedural", "difficulty": "medium"},
    "d14": {"sources": ["manuel-exacor.pdf", "Présentation_copies_atypiques_VF.pptx"], "lang": "fr", "type": "analytical", "difficulty": "hard"},
    "d15": {"sources": ["B2_Grille_PO.pdf", "PO_B2.pdf"], "lang": "fr", "type": "factoid", "difficulty": "easy"},
    "d16": {"sources": ["manuel-exacor.pdf", "Diaporama_rôle_EC_VF.pptx"], "lang": "fr", "type": "definition", "difficulty": "easy"},
    "d17": {"sources": ["manuel-exacor.pdf"], "lang": "fr", "type": "analytical", "difficulty": "medium"},
    "d18": {"sources": ["manuel-exacor.pdf"], "lang": "tr", "type": "procedural", "difficulty": "hard"},
}

# Held-out: ~30% stratified across categories (6 of 18 originals)
HELD_OUT_OLD = {"d04", "d07", "d11", "d13", "d16", "d18"}

migrated = []
for q in old:
    rid = q["id"]
    r = REMAP[rid]
    nq = dict(q)  # keep original question + ground_truth (RAGAS back-compat)
    nq["language"] = r["lang"]
    nq["type"] = r["type"]
    nq["difficulty"] = r["difficulty"]
    nq["expected_sources"] = r["sources"]
    nq["answerable"] = True
    nq["split"] = "held_out" if rid in HELD_OUT_OLD else "dev"
    nq["validation_status"] = "UNVALIDATED_NEEDS_DELF_EXPERT"
    migrated.append(nq)


def Q(id, category, level, skill, lang, typ, diff, qtr, qfr, atr, afr, sources, split):
    return {
        "id": id,
        "category": category,
        "level": level,
        "skill": skill,
        # `question`/`ground_truth` mirror the item's primary language for RAGAS back-compat
        "question": qtr if lang == "tr" else qfr,
        "ground_truth": atr if lang == "tr" else afr,
        "language": lang,
        "type": typ,
        "difficulty": diff,
        "question_tr": qtr,
        "question_fr": qfr,
        "expected_answer_tr": atr,
        "expected_answer_fr": afr,
        "expected_sources": sources,
        "answerable": True,
        "split": split,
        "validation_status": "UNVALIDATED_NEEDS_DELF_EXPERT",
    }


new = [
    # --- CECRL / niveaux (descripteur) ----------------------------------------
    Q("d19", "descripteur", "general", "general", "fr", "definition", "easy",
      "CECRL kaç seviyeden oluşur ve bunlar nasıl gruplanır?",
      "Combien de niveaux compte le CECRL et comment sont-ils regroupés ?",
      "CECRL altı seviyeden oluşur: A1 ve A2 (utilisateur élémentaire / temel kullanıcı), B1 ve B2 (utilisateur indépendant / bağımsız kullanıcı), C1 ve C2 (utilisateur expérimenté / deneyimli kullanıcı). Her seviye yetkinlik descripteur'leri ile tanımlanır.",
      "Le CECRL compte six niveaux : A1 et A2 (utilisateur élémentaire), B1 et B2 (utilisateur indépendant), C1 et C2 (utilisateur expérimenté). Chaque niveau est défini par des descripteurs de compétences.",
      ["Descripteurs_CECRL_A1_B2_VF_a_donner.pdf", "Echelle globale.pdf"], "dev"),

    Q("d20", "presentation", "general", "general", "tr", "factoid", "easy",
      "DELF ve DALF diplomaları hangi CECRL seviyelerini kapsar?",
      "Quels niveaux du CECRL couvrent respectivement le DELF et le DALF ?",
      "DELF (Diplôme d'études en langue française) A1'den B2'ye kadar olan seviyeleri kapsar; DALF (Diplôme approfondi de langue française) ise C1 ve C2 seviyelerini kapsar.",
      "Le DELF (Diplôme d'études en langue française) couvre les niveaux A1 à B2 ; le DALF (Diplôme approfondi de langue française) couvre les niveaux C1 et C2.",
      ["Dispositif_DELF-DALF_v2022.pptx"], "held_out"),

    Q("d21", "presentation", "general", "general", "fr", "factoid", "medium",
      "Bir DELF/DALF diplomasını almak için gereken toplam ve epreuve başına minimum puan nedir?",
      "Quelle est la note totale requise et la note minimale par épreuve pour obtenir un diplôme DELF/DALF ?",
      "Toplam not 100 üzerindendir ve başarı eşiği 50/100'dür. Epreuvelerden birinde 5/25'in altında bir not eleyicidir (éliminatoire); bu durumda toplam 50'ye ulaşsa bile aday başarısız sayılır.",
      "La note totale est sur 100 et le seuil de réussite est fixé à 50/100. Une note inférieure à 5/25 à l'une des épreuves est éliminatoire : dans ce cas le candidat est ajourné même si le total atteint 50.",
      ["manuel-exacor.pdf"], "dev"),

    Q("d22", "presentation", "general", "general", "tr", "factoid", "easy",
      "Her DELF/DALF sınavı hangi dört epreuveden oluşur?",
      "De quelles quatre épreuves se compose chaque examen DELF/DALF ?",
      "Her diploma her bir dilsel beceriye karşılık gelen dört epreuveden oluşur: compréhension de l'oral (sözlü anlama), compréhension des écrits (yazılı anlama), production écrite (yazılı üretim) ve production orale (sözlü üretim).",
      "Chaque diplôme se compose de quatre épreuves, une par compétence langagière : compréhension de l'oral, compréhension des écrits, production écrite et production orale.",
      ["Dispositif_DELF-DALF_v2022.pptx", "Tableau_epreuves_DD_AS.pdf"], "dev"),

    # --- Copies atypiques (analytical / definition) ---------------------------
    Q("d23", "copie", "general", "PE", "fr", "analytical", "medium",
      "Adayın talimata kısmen uyduğu (non-respect partiel de la consigne) bir kopya hors-sujet sayılır mı?",
      "Une copie où le candidat ne respecte que partiellement la consigne est-elle un hors-sujet ?",
      "Hayır. Aday bir söz edimini (acte de parole) atlarsa bu, talimata kısmi uyumsuzluktur ve bir hors-sujet durumu DEĞİLDİR.",
      "Non. Si le candidat oublie un acte de parole, il s'agit d'un non-respect partiel de la consigne ; ce n'est pas un cas de hors-sujet.",
      ["Présentation_copies_atypiques_VF.pptx"], "dev"),

    Q("d24", "copie", "general", "PE", "tr", "definition", "medium",
      "« Copie atypique » nedir? Hangi farklı hors-sujet türleri vardır?",
      "Qu'est-ce qu'une « copie atypique » ? Quels sont les différents types de hors-sujet ?",
      "Copie atypique, olağandışı/sıra dışı bir kopyadır. Farklı hors-sujet türleri arasında talimata kısmi uyumsuzluk (hors-sujet değildir), tematik hors-sujet ve söylemsel (discursif) hors-sujet bulunur; tam hors-sujet hem tematik hem söylemsel olandır.",
      "Une copie atypique sort de l'ordinaire. Parmi les types figurent le non-respect partiel de la consigne (qui n'est pas un hors-sujet), le hors-sujet thématique et le hors-sujet discursif ; le hors-sujet complet est à la fois thématique et discursif.",
      ["Présentation_copies_atypiques_VF.pptx"], "held_out"),

    Q("d25", "descripteur", "general", "general", "fr", "definition", "easy",
      "Bir « descripteur » nedir ve değerlendirmede ne işe yarar?",
      "Qu'est-ce qu'un « descripteur » et à quoi sert-il dans l'évaluation ?",
      "Descripteur, bir yetkinlik seviyesini tanımlayan bir ifadedir; düzeltici, adayın üretimini ilgili kritere göre konumlandırmak için descripteur'leri kullanır.",
      "Un descripteur est un énoncé décrivant un niveau de compétence ; le correcteur s'en sert pour positionner la production du candidat sur le critère concerné.",
      ["Descripteurs_CECRL_A1_B2_VF_a_donner.pdf", "B1_Descripteurs_PO.pdf"], "dev"),

    Q("d26", "grille", "general", "PE", "tr", "definition", "easy",
      "Yazılı üretim grilinde hangi beş değerlendirme kriteri ortak olarak yer alır?",
      "Quels sont les cinq critères d'évaluation communs aux grilles de production écrite ?",
      "Yazılı üretim grillerinde ortak beş kriter şunlardır: Réalisation de la tâche (görevin gerçekleştirilmesi), Cohérence et cohésion, Adéquation sociolinguistique, Lexique ve Morphosyntaxe.",
      "Les cinq critères communs aux grilles de production écrite sont : Réalisation de la tâche, Cohérence et cohésion, Adéquation sociolinguistique, Lexique et Morphosyntaxe.",
      ["B1_Grille_PE.pdf", "B2_Grille_PE.pdf"], "dev"),

    Q("d27", "copie", "general", "PE", "fr", "analytical", "medium",
      "Aday talimattan çok daha fazla (örn. 40 yerine 75 kelime) yazarsa cezalandırılır mı?",
      "Le candidat est-il pénalisé s'il écrit beaucoup plus que demandé (par ex. 75 mots au lieu de 40) ?",
      "Hayır. İstenenden fazla yazmak, belirli bir kriterde cezalandırılmaz ve değerlendirilmez (C1 synthèse hariç).",
      "Non. Écrire plus que demandé n'est pas sanctionné ni évalué dans un critère spécifique (hormis dans la synthèse du C1).",
      ["Introduction_critères_corriges.docx"], "dev"),

    Q("d28", "copie", "general", "PE", "tr", "analytical", "medium",
      "Bir aday istenen 40 kelime yerine yalnızca 18 kelime yazarsa kopya nasıl puanlanır?",
      "Comment note-t-on une copie où le candidat écrit seulement 18 mots au lieu des 40 demandés ?",
      "Bu durumda « Manque de matière évaluable » (değerlendirilebilir içerik eksikliği) kutusu « anomalies » bölümünde işaretlenir ve kopya 0 alır.",
      "On coche la case « Manque de matière évaluable » dans la section « anomalies » et la copie obtient 0.",
      ["Introduction_critères_corriges.docx"], "held_out"),

    Q("d29", "methodologie", "general", "PE", "fr", "analytical", "hard",
      "Aday yalnızca tek bir etkinlikten söz etmesine rağmen birkaç etkinlik betimlemesi gerekiyorsa bu hangi kriterde değerlendirilir?",
      "Si le candidat doit décrire plusieurs activités mais n'en mentionne qu'une, dans quel critère cela est-il évalué ?",
      "Bu eksiklik « Réalisation de la tâche » (görevin gerçekleştirilmesi) kriterinde değerlendirilir; çünkü görevin gereklerinin tümü karşılanmamıştır.",
      "Cela est évalué dans le critère « Réalisation de la tâche », car les exigences de la tâche ne sont pas toutes satisfaites.",
      ["Introduction_critères_corriges.docx", "manuel-exacor.pdf"], "dev"),

    Q("d30", "grille", "A1", "PE", "tr", "factoid", "easy",
      "DELF A1 yazılı üretim grilinde hangi anomaliler (anomalies) işaretlenebilir?",
      "Quelles anomalies peut-on cocher dans la grille de production écrite du DELF A1 ?",
      "DELF A1 yazılı üretim grilinde, üretimde anomali varsa ilgili kutular işaretlenir; bunlar arasında hors-sujet ve « Manque de matière évaluable » (değerlendirilebilir içerik eksikliği) yer alır.",
      "Dans la grille de production écrite du DELF A1, si la production contient des anomalies, on coche les cases correspondantes, parmi lesquelles le hors-sujet et le « Manque de matière évaluable ».",
      ["A1_Grille_PE.pdf"], "dev"),

    # --- Rôle de l'examinateur-correcteur (presentation / definition) ---------
    Q("d31", "presentation", "general", "general", "fr", "definition", "medium",
      "Examinateur-correcteur'ün rolü neden önemlidir?",
      "Pourquoi le rôle de l'examinateur-correcteur est-il primordial ?",
      "Examinateur-correcteur'ün rolü, adaya iyi bir performans gösterme şansını vermek, diplomaların güvenilirliğini (fiabilité) ve değerlendirmelerin eşitliğini (équité) güvence altına almak için önemlidir.",
      "Le rôle de l'examinateur-correcteur est primordial pour donner au candidat toutes les chances de réaliser une bonne performance, pour garantir la fiabilité des diplômes et l'équité des évaluations.",
      ["Diaporama_rôle_EC_VF.pptx"], "held_out"),

    Q("d32", "double_correction", "general", "general", "tr", "procedural", "medium",
      "Düzeltici, her egzersizin notunu kopyanın neresine aktarır?",
      "Où le correcteur reporte-t-il la note de chaque exercice sur la copie ?",
      "Düzeltici puanları toplar ve her egzersizin notunu kopyanın kapak sayfasına (page de garde) aktarır.",
      "Le correcteur additionne les points et reporte la note de chaque exercice sur la page de garde de la copie.",
      ["manuel-exacor.pdf"], "dev"),

    Q("d33", "double_correction", "general", "general", "fr", "procedural", "medium",
      "İki düzeltici arasında basit bir hesap/barem hatası olduğunda ikinci düzeltici ne yapar?",
      "Que fait le correcteur n°2 en cas d'erreur de calcul ou de barème (et non de désaccord) ?",
      "İkinci düzeltici, sayım/hesap hatasını veya açık barem hatasını çizerek düzeltir, puanı ve toplam notu yeniden hesaplar; bu durumda harmonizasyon gerekmez.",
      "Le correcteur n°2 barre et corrige l'erreur de comptage ou l'erreur manifeste de barème, puis recalcule la note et le total ; aucune harmonisation n'est nécessaire dans ce cas.",
      ["manuel-exacor.pdf"], "dev"),

    # --- Procédure / dispositif (procedural / factoid) ------------------------
    Q("d34", "procedure", "general", "general", "tr", "factoid", "medium",
      "DELF/DALF'ın yönetimi nasıl düzenlenmiştir?",
      "Comment est organisée la gestion du DELF/DALF ?",
      "DELF/DALF'ın yönetimi hem merkezî hem de yerel (décentralisée) yapıdadır; bunlar Millî Eğitim Bakanlığı'nın devlet diplomalarıdır ve Fransa'da rektörlükler (rectorats) dağıtım, sınav merkezlerinin takibi ve prosedürlere uyumdan sorumludur.",
      "La gestion du DELF/DALF est à la fois centralisée et décentralisée ; ce sont des diplômes d'État du ministère de l'Éducation nationale et, en France, les rectorats sont responsables de leur diffusion, du suivi des centres d'examen et du respect des procédures.",
      ["Dispositif_DELF-DALF_v2022.pptx"], "dev"),

    Q("d35", "double_correction", "general", "general", "fr", "procedural", "hard",
      "İki düzeltici arasında bir görüş ayrılığı (désaccord) olduğunda izlenen prosedür nedir?",
      "Quelle est la procédure suivie en cas de désaccord entre les deux correcteurs ?",
      "İlgili sorular, kolayca bulunabilmesi için kurşun kalemle bir soru işareti konularak işaretlenir; ardından iki düzeltici notlarını uyumlamak için bir concertation (görüşme/harmonizasyon) yapar.",
      "Les questions concernées sont signalées au crayon de papier par un point d'interrogation afin de les repérer facilement, puis une concertation entre les deux correcteurs est organisée pour harmoniser leurs notations.",
      ["manuel-exacor.pdf"], "held_out"),

    # --- Grilles par niveau (factoid / definition) ----------------------------
    Q("d36", "grille", "B1", "PE", "fr", "factoid", "medium",
      "DELF B1 yazılı üretim grilinde « Réalisation de la tâche » kriterindeki puan basamakları nedir?",
      "Quels sont les échelons de notation du critère « Réalisation de la tâche » dans la grille de production écrite du DELF B1 ?",
      "DELF B1 yazılı üretim grilinde « Réalisation de la tâche » kriteri 0, 1, 3 ve 5 puan basamaklarıyla derecelendirilir (Non répondu / En dessous du niveau ciblé / B1 / B1+).",
      "Dans la grille de production écrite du DELF B1, le critère « Réalisation de la tâche » est noté selon les échelons 0, 1, 3 et 5 (Non répondu / En dessous du niveau ciblé / B1 / B1+).",
      ["B1_Grille_PE.pdf"], "dev"),

    Q("d37", "grille", "B1", "PO", "tr", "factoid", "medium",
      "DELF B1 sözlü üretim epreuvesi hangi üç görevden (tâche) oluşur?",
      "De quelles trois tâches se compose l'épreuve de production orale du DELF B1 ?",
      "DELF B1 sözlü üretim grili üç görev içerir: entretien dirigé (yönlendirilmiş görüşme, 2-3 dk), exercice en interaction (etkileşimli alıştırma, 3-4 dk) ve expression d'un point de vue (görüş bildirme, 5-7 dk).",
      "La grille de production orale du DELF B1 comporte trois tâches : l'entretien dirigé (2 à 3 minutes), l'exercice en interaction (3 à 4 minutes) et l'expression d'un point de vue (5 à 7 minutes).",
      ["B1_Grille_PO.pdf", "PO_B1.pdf"], "dev"),

    Q("d38", "grille", "A1", "PO", "fr", "definition", "easy",
      "DELF A1 sözlü üretim grili hangi üç görevi (tâche) içerir?",
      "Quelles sont les trois tâches de la grille de production orale du DELF A1 ?",
      "DELF A1 sözlü üretim grili üç görev içerir: entretien dirigé (yönlendirilmiş görüşme, ~1 dk), échange d'informations (bilgi alışverişi, ~2 dk) ve dialogue simulé (canlandırılmış diyalog, ~2 dk).",
      "La grille de production orale du DELF A1 comporte trois tâches : l'entretien dirigé (environ 1 minute), l'échange d'informations (environ 2 minutes) et le dialogue simulé (environ 2 minutes).",
      ["A1_Grille_PO.pdf", "PO_A1.pdf"], "dev"),

    Q("d39", "grille", "A2", "PO", "fr", "factoid", "medium",
      "DELF A2 sözlü üretim epreuvesi hangi üç bölümden oluşur?",
      "De quelles trois parties se compose l'épreuve de production orale du DELF A2 ?",
      "DELF A2 sözlü üretim grili üç görev içerir: entretien dirigé (yönlendirilmiş görüşme, ~1 dk 30), monologue suivi (sürekli monolog, ~2 dk) ve exercice en interaction (etkileşimli alıştırma, 3-4 dk).",
      "La grille de production orale du DELF A2 comporte trois tâches : l'entretien dirigé (environ 1 min 30), le monologue suivi (environ 2 minutes) et l'exercice en interaction (3 à 4 minutes).",
      ["A2_Grille_PO.pdf", "PO_A2.pdf"], "held_out"),

    Q("d40", "grille", "C1", "PE", "tr", "factoid", "hard",
      "DALF C1 yazılı üretim epreuvesi hangi iki alıştırmadan oluşur?",
      "De quels deux exercices se compose l'épreuve de production écrite du DALF C1 ?",
      "DALF C1 yazılı üretim iki alıştırmadan oluşur: synthèse de documents (belge sentezi) ve essai argumenté (gerekçeli deneme).",
      "L'épreuve de production écrite du DALF C1 se compose de deux exercices : une synthèse de documents et un essai argumenté.",
      ["dalf-c1sujet-demo-candidat-coll.pdf"], "dev"),

    # --- Descripteurs (definition / analytical) -------------------------------
    Q("d41", "descripteur", "B1", "general", "fr", "definition", "easy",
      "CECRL küresel ölçeğine (échelle globale) göre B1 seviyesindeki bir kullanıcı neler yapabilir?",
      "Selon l'échelle globale du CECRL, que peut faire un utilisateur de niveau B1 ?",
      "B1 kullanıcısı, açık ve standart bir dil kullanıldığında ve iş, okul, boş zaman gibi tanıdık konularda ana noktaları anlayabilir; seyahatte karşılaşılan çoğu durumda kendini idare edebilir ve tanıdık konularda basit ve tutarlı bir söylem üretebilir.",
      "L'utilisateur B1 peut comprendre les points essentiels quand un langage clair et standard est utilisé et s'il s'agit de choses familières (travail, école, loisirs) ; il peut se débrouiller dans la plupart des situations de voyage et produire un discours simple et cohérent sur des sujets familiers.",
      ["Echelle globale.pdf"], "dev"),

    Q("d42", "descripteur", "A1", "PO", "tr", "definition", "medium",
      "DELF A1 sözlü üretiminde « Réalisation de la tâche » için hedef seviyedeki (au niveau ciblé) descripteur ne der?",
      "Que dit le descripteur « au niveau ciblé » pour la « Réalisation de la tâche » en production orale du DELF A1 ?",
      "Hedef seviyede aday kendini tanıtabilir ve kimliği ile yakın çevresi hakkında basit terimlerle, art arda sıralanmış (juxtaposées) fikirler biçiminde konuşabilir; ancak basit ve öngörülebilir bazı sorular anlaşılmayabilir.",
      "Au niveau ciblé, le candidat peut se présenter et parler en termes simples et sous forme d'idées juxtaposées de son identité et de son environnement immédiat ; certaines questions simples et prévisibles peuvent ne pas être comprises.",
      ["A1_Descripteurs_PO.pdf"], "dev"),

    Q("d43", "descripteur", "B2", "PO", "fr", "analytical", "medium",
      "DELF B2 sözlü üretiminde « Réalisation de la tâche » açısından adaydan ne beklenir?",
      "Qu'attend-on du candidat pour la « Réalisation de la tâche » en production orale du DELF B2 ?",
      "B2 seviyesinde aday, konuda ortaya atılan tematiği belirleyebilmeli ve problematiği bir giriş ile sunabilmeli; görüşünü, argümanlara dayanarak ve lehte ya da aleyhte gerekçeler getirerek bildirebilmelidir.",
      "Au niveau B2, le candidat peut identifier la thématique soulevée par le sujet et présenter la problématique au moyen d'une introduction ; il peut donner son point de vue en s'appuyant sur des arguments et en apportant des justifications pour ou contre.",
      ["B2_Descripteurs_PO.pdf"], "held_out"),

    Q("d44", "methodologie", "C1", "PO", "fr", "procedural", "medium",
      "DALF C1 sözlü üretim sınavının exposé bölümü için aday ne kadar hazırlık süresi alır ve ne kullanabilir?",
      "De combien de temps de préparation dispose le candidat pour l'exposé de la production orale du DALF C1, et que peut-il utiliser ?",
      "Aday exposé için 60 dakika hazırlık süresi alır ve sınav salonunda bulunan tek dilli (monolingue) bir Fransızca sözlük kullanabilir; kendi kişisel sözlüğünü getiremez.",
      "Le candidat dispose de 60 minutes de préparation pour l'exposé et peut utiliser un dictionnaire monolingue français disponible dans la salle d'examen ; il ne peut pas apporter son dictionnaire personnel.",
      ["DALF C1-PO-Méthodologie-VD.pdf"], "dev"),

    Q("d45", "methodologie", "general", "PE", "tr", "procedural", "easy",
      "Üretilmesi gereken minimum kelime sayısına uyulmaması nasıl bir sonuç doğurur?",
      "Quelle conséquence entraîne le non-respect du nombre minimal de mots à produire ?",
      "Minimum kelime sayısına uyulmaması, puan açısından bir yaptırıma (sanction) yol açabilir; bunun göstergeleri değerlendirme grilinin « anomalies » bölümünde belirtilir.",
      "Le non-respect du nombre minimal de mots peut entraîner une sanction sur les points ; les indications figurent dans la section « anomalies » de la grille d'évaluation.",
      ["manuel-exacor.pdf"], "dev"),

    # --- Niveaux / variantes / tableaux (factoid) -----------------------------
    Q("d46", "presentation", "general", "general", "fr", "factoid", "easy",
      "DELF'in hangi farklı versiyonları (public) vardır?",
      "Quelles sont les différentes versions (publics) du DELF ?",
      "DELF'in farklı versiyonları vardır: DELF tout public (yetişkinler), DELF junior ve DELF scolaire (ergenler) ve DELF Prim (çocuklar); her versiyon hedef kitleye uyarlanmıştır.",
      "Le DELF existe en plusieurs versions : le DELF tout public (adultes), le DELF junior et le DELF scolaire (adolescents) et le DELF Prim (enfants) ; chaque version est adaptée au public visé.",
      ["Dispositif_DELF-DALF_v2022.pptx"], "dev"),

    Q("d47", "presentation", "A1", "general", "tr", "factoid", "medium",
      "Épreuvelerin tablosuna göre DELF A1 sözlü epreuvesi kaç dakika sürer ve kaç bölümden oluşur?",
      "D'après le tableau des épreuves, combien de temps dure l'épreuve orale du DELF A1 et combien de parties comporte-t-elle ?",
      "DELF A1 sözlü epreuvesi jüriyle 5 ila 7 dakikalık bir görüşmedir ve üç bölümden oluşur: entretien dirigé, échange d'informations ve dialogue simulé.",
      "L'épreuve orale du DELF A1 est un entretien de 5 à 7 minutes avec le jury, en trois parties : l'entretien dirigé, l'échange d'informations et le dialogue simulé.",
      ["Tableau_epreuves_DD_AS.pdf"], "held_out"),

    Q("d48", "descripteur", "general", "PE", "fr", "definition", "medium",
      "Niveaux_CECRL_PE alıştırmasında bir düzeltici yazılı üretimleri nasıl değerlendirir?",
      "Dans l'exercice d'évaluation des productions écrites par niveaux, comment le correcteur procède-t-il ?",
      "Düzeltici her üretime bir seviye (A1'den B2'ye) atar, ardından seviyeleri ilgili descripteur'lere bağlar ve seçimini gerekçelendirir.",
      "Le correcteur attribue un niveau (de A1 à B2) à chaque production, puis relie les niveaux aux descripteurs correspondants en justifiant son choix.",
      ["Niveaux_CECRL_PE_V2.pdf"], "dev"),

    Q("d49", "methodologie", "C1", "PO", "tr", "factoid", "medium",
      "DALF C1 sözlü üretim epreuvesi hangi iki bölümden oluşur?",
      "De quelles deux parties se compose l'épreuve de production orale du DALF C1 ?",
      "DALF C1 sözlü üretim iki bölümden oluşur: gerekçeli bir görüşün savunulduğu bir exposé (sunum) ve ardından jüriyle bir tartışma (interaction/débat).",
      "L'épreuve de production orale du DALF C1 est constituée de deux parties : un exposé (défense d'un point de vue argumenté) suivi d'un débat (interaction) avec le jury.",
      ["DALF C1-PO-Méthodologie-VD.pdf"], "dev"),

    Q("d50", "descripteur", "A2", "general", "fr", "definition", "easy",
      "Anahtar kelimelere (mots-clés) göre A2 seviyesinin başlıca özellikleri nelerdir?",
      "D'après les mots-clés, quelles sont les principales caractéristiques du niveau A2 ?",
      "A2 (intermédiaire ou de survie) seviyesi şu anahtar kelimelerle tanımlanır: günlük yaşam etkinlikleri ve acil ihtiyaçlar, genişletilmiş kişisel temalar, basit ve yalıtık cümleler, açık ve duyulabilir bir dil ile selamlaşma ve bilgi isteme gibi dil işlevleri.",
      "Le niveau A2 (intermédiaire ou de survie) est caractérisé par : les activités de la vie quotidienne et les besoins immédiats, les thèmes personnels élargis, des phrases simples et isolées, un langage clair et audible, et des fonctions du langage comme saluer et demander des informations.",
      ["niveaux_mots_cles.pdf"], "dev"),
]

questions = migrated + new
assert len(questions) == 50, len(questions)

# Language + split tallies
lang = {"tr": 0, "fr": 0}
split = {"dev": 0, "held_out": 0}
for q in questions:
    lang[q["language"]] += 1
    split[q["split"]] += 1

doc = {
    "metadata": {
        "version": "2.0",
        "created": "2026-05-29",
        "description": (
            "DELF/DALF sınavcı-düzeltici (examiner-correcteur) bilingual değerlendirme seti. "
            "Sorular TR + FR; cross-lingual retrieval yolunu (TR sorgu → FR korpus) ve same-language "
            "(FR) yolunu test eder. v2.0: expected_sources alanları, ChromaDB'den okunan gerçek "
            "korpus dosya adlarına (source_filename metadata) yeniden eşlendi."
        ),
        "total_questions": len(questions),
        "language_split": lang,
        "split_distribution": split,
        "schema_note": (
            "Her soruda hem `question`/`ground_truth` (birincil dil — RAGAS geriye uyumluluğu) "
            "hem de `question_tr`/`question_fr`/`expected_answer_tr`/`expected_answer_fr` bulunur. "
            "`expected_sources` recall ölçümü için kullanılır ve YALNIZCA gerçek korpus dosya adları içerir. "
            "`split` ∈ {dev, held_out} (~70/30 stratified). `type` ∈ {factoid, definition, analytical, procedural}."
        ),
        "validation": (
            "PROVISIONAL. Tüm expected_sources değerleri ChromaDB child metadata'sından okunan içeriğe göre "
            "OTOMATİK olarak gerçek korpus dosyalarına eşlendi (eski v1.0 dosya adları kurgusaldı). Referans "
            "cevaplar korpus chunk'larından temellendirildi ANCAK hiçbiri DELF uzmanı tarafından doğrulanmadı. "
            "Her soru validation_status='UNVALIDATED_NEEDS_DELF_EXPERT' taşır. M3/M4 kapıları, bu set bir DELF "
            "uzmanı tarafından doğrulanana kadar SERTİFİKALI sayılmamalıdır."
        ),
        "coverage_note": "A1-C2 seviyeleri; PE/PO/CE/CO becerileri; grille/descripteur/methodologie/copie/glossaire/guide/procedure/presentation doc_type/kategorileri kapsanır.",
    },
    "questions": questions,
}

(BASE / "delf_questions.json").write_text(
    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("total", len(questions), "lang", lang, "split", split)
