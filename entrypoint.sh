#!/bin/sh
# Konteyner giriş noktası — Streamlit'ten ÖNCE korpusu hazırlar (Phase 1).
#
# python -m src.bootstrap:
#   - PERSISTENCE_BACKEND=gcs ise korpusu GCS'ten yerel diske indirir/senkronize eder,
#   - hazırlık kontrolü yapar,
#   - REQUIRE_CORPUS_ON_STARTUP=true iken korpus hazır değilse sıfırdan farklı kodla çıkar.
# 'set -e' sayesinde bootstrap başarısız olursa konteyner başlamaz (Cloud Run
# revizyonu sağlıksız sayılır ve trafik almaz). REQUIRE kapalıyken bootstrap 0
# döner; Streamlit açılır ve uygulama net bir "korpus kullanılamıyor" banner'ı gösterir.
set -e

python -m src.bootstrap

exec streamlit run app.py \
  --server.port="${PORT:-8080}" \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false
