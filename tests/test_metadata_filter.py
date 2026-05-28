"""
Metadata ön-filtreleme birim testleri.

Saf fonksiyon testleri — ChromaDB veya embedding gerektirmez. where-clause
biçimi, şema doğrulaması ve muhafazakar sorgu tespiti (asla tahmin etmez)
doğrulanır.
"""

from __future__ import annotations

from src.metadata_filter import (
    build_where_clause,
    detect_filters,
    matches_metadata,
    merge_where,
)


class TestBuildWhereClause:
    def test_tek_alan_sade_eq(self):
        assert build_where_clause(level="B2") == {"level": {"$eq": "B2"}}

    def test_cok_alan_and_ile_birlesir(self):
        where = build_where_clause(level="B2", skill="PO")
        assert where == {"$and": [{"level": {"$eq": "B2"}}, {"skill": {"$eq": "PO"}}]}

    def test_authoritative_only_kosul_ekler(self):
        where = build_where_clause(doc_type="grille", authoritative_only=True)
        assert {"is_authoritative": {"$eq": True}} in where["$and"]

    def test_gecersiz_deger_yok_sayilir(self):
        # Şema literal'inde olmayan değerler sessizce atlanır.
        assert build_where_clause(level="ZZ") is None
        assert build_where_clause(skill="XX") is None

    def test_gecerli_ve_gecersiz_karisik(self):
        # Geçerli olan korunur, geçersiz atlanır → tek koşul kalır.
        assert build_where_clause(level="B2", skill="XX") == {"level": {"$eq": "B2"}}

    def test_hicbir_kosul_yoksa_none(self):
        assert build_where_clause() is None


class TestDetectFilters:
    def test_kesin_seviye_ve_beceri(self):
        assert detect_filters("B2 PO grilleri göster") == {"level": "B2", "skill": "PO"}

    def test_yalnizca_seviye(self):
        assert detect_filters("A1 üretim nasıl değerlendirilir") == {"level": "A1"}

    def test_belirsiz_birden_cok_seviye_atlanir(self):
        # B1 ve B2 birlikte geçiyor → tutarsız, seviye filtresi uygulanmaz.
        assert detect_filters("B1 ve B2 arasındaki fark nedir") == {}

    def test_kucuk_harf_beceri_eslesmez(self):
        # "po" küçük harf — Fransızca sözcüklerle karışmasın diye eşleşmez.
        assert "skill" not in detect_filters("la production po orale")

    def test_hicbir_token_yoksa_bos(self):
        assert detect_filters("grille nedir") == {}


class TestMergeWhere:
    def test_none_atlanir(self):
        assert merge_where(None, {"level": {"$eq": "B2"}}) == {"level": {"$eq": "B2"}}

    def test_hepsi_none_ise_none(self):
        assert merge_where(None, None) is None

    def test_iki_clause_and_altinda(self):
        merged = merge_where({"source": "x"}, build_where_clause(level="B2"))
        assert merged == {"$and": [{"source": "x"}, {"level": {"$eq": "B2"}}]}

    def test_ic_ice_and_duzlestirilir(self):
        a = build_where_clause(level="B2", skill="PO")  # {"$and": [...]}
        merged = merge_where(a, {"source": "x"})
        # $and düzleştirilmeli — iç içe $and oluşmamalı.
        assert merged["$and"] == [
            {"level": {"$eq": "B2"}},
            {"skill": {"$eq": "PO"}},
            {"source": "x"},
        ]


class TestMatchesMetadata:
    def test_none_where_her_zaman_eslesir(self):
        assert matches_metadata({"level": "A1"}, None) is True

    def test_eq_operatoru(self):
        where = build_where_clause(level="B2")
        assert matches_metadata({"level": "B2"}, where) is True
        assert matches_metadata({"level": "C1"}, where) is False

    def test_and_tum_kosullar(self):
        where = build_where_clause(level="B2", skill="PO")
        assert matches_metadata({"level": "B2", "skill": "PO"}, where) is True
        assert matches_metadata({"level": "B2", "skill": "PE"}, where) is False

    def test_duz_esitlik_bicimi(self):
        assert matches_metadata({"source": "x"}, {"source": "x"}) is True
        assert matches_metadata({"source": "y"}, {"source": "x"}) is False
