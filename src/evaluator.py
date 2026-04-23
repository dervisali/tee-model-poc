"""
Evaluator module stub for TEE-Model POC — Faz 2 placeholder.

These functions are intentionally unimplemented for the POC phase.
Full implementation requires LMS/Google Forms integration and
comparative pre/post-test analytics.
"""


def run_pre_test(user_id: str) -> dict:
    """
    TODO Faz 2: Eğitim öncesi ön test uygula ve sonuçları kaydet.
    Google Forms veya basit LMS (Moodle) entegrasyonu gerektirir.

    Parameters
    ----------
    user_id : str
        Eğitimi alan personelin benzersiz kimliği.

    Returns
    -------
    dict
        Ön test sonuçları (soru bazlı doğru/yanlış ve toplam puan).
    """
    raise NotImplementedError


def run_post_test(user_id: str) -> dict:
    """
    TODO Faz 2: Eğitim sonrası son test uygula.
    Ön test sonuçlarıyla karşılaştırmalı analiz yapılacak.

    Parameters
    ----------
    user_id : str
        Eğitimi alan personelin benzersiz kimliği.

    Returns
    -------
    dict
        Son test sonuçları (soru bazlı doğru/yanlış ve toplam puan).
    """
    raise NotImplementedError


def analyze_results(pre: dict, post: dict) -> dict:
    """
    TODO Faz 2: Hata oranı analizi, süre analizi ve öğrenme kazanımı raporunu üret.
    Çıktı Excel/Google Sheets formatında dışa aktarılabilir olacak.

    Parameters
    ----------
    pre : dict
        run_pre_test() çıktısı.
    post : dict
        run_post_test() çıktısı.

    Returns
    -------
    dict
        Karşılaştırmalı analiz raporu (kazanım oranı, zayıf noktalar, öneri).
    """
    raise NotImplementedError
