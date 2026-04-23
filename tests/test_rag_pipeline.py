"""
tests/test_rag_pipeline.py

Critical path tests for the TEE-Model RAG pipeline.
Tests each stage in isolation (Parts 1–2), then as an integrated system (Part 3).

Run with:
    cd /Users/dervis/Desktop/tee_project/tee-model-poc
    python -m unittest tests/test_rag_pipeline.py -v
"""

import sys
import os
import json
import time
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: project root must be on sys.path before any src.* imports
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

import chromadb
from src.anonymizer import anonymize_text
from src.ingestion import (
    _split_into_chunks,
    _get_genai_client,
    _embed_chunk,
    _get_chroma_collection,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    COLLECTION_NAME,
    CHROMA_DIR,
    insert_new_document,
)
from src.retrieval import retrieve_context, _embed_query
from src.generators import generate_process_map

CHROMA_DIR_STR = str(CHROMA_DIR)


# ===========================================================================
# TEST 1: Anonymizer correctness
# ===========================================================================

class TestAnonymizer(unittest.TestCase):
    """
    Test 1: Verify KVKK anonymizer correctly masks all four PII types
    and leaves process-relevant numbers untouched.
    """

    # A single text that contains exactly one of each PII type, plus a
    # process-relevant percentage that must NOT be masked.
    SAMPLE_TEXT = (
        "Maaş mutemetliği görevini yürüten kişinin adı Ayşe Nur Demir, "
        "TC kimlik numarası 23456789012, "
        "banka IBAN numarası TR330006100519786457841326 ve "
        "telefon numarası 05321234567'dir. "
        "Maaş artışı %15 vergi dilimi üzerinden hesaplanır."
    )

    def setUp(self):
        self.result = anonymize_text(self.SAMPLE_TEXT)
        self.anon = self.result["anonymized_text"]
        self.log = self.result["mask_log"]

    def test_name_masked(self):
        self.assertIn("[İSİM]", self.anon,
            f"Turkish name NOT masked.\nAnonymized text: {self.anon}")

    def test_tc_kimlik_masked(self):
        self.assertIn("[TC-KİMLİK]", self.anon,
            f"TC Kimlik NOT masked.\nAnonymized text: {self.anon}")

    def test_iban_masked(self):
        self.assertIn("[IBAN]", self.anon,
            f"IBAN NOT masked.\nAnonymized text: {self.anon}")

    def test_phone_masked(self):
        self.assertIn("[TELEFON]", self.anon,
            f"Phone NOT masked.\nAnonymized text: {self.anon}")

    def test_mask_log_has_exactly_4_entries(self):
        self.assertEqual(len(self.log), 4,
            f"Expected 4 mask entries, got {len(self.log)}.\n"
            f"Log contents: {self.log}\n"
            f"Anonymized text: {self.anon}")

    def test_process_relevant_number_not_masked(self):
        self.assertIn("%15 vergi", self.anon,
            f"'%15 vergi' was incorrectly masked.\nAnonymized text: {self.anon}")


# ===========================================================================
# TEST 2: Chunker behaviour
# ===========================================================================

class TestChunker(unittest.TestCase):
    """
    Test 2: Verify the chunking strategy handles paragraphs, long text,
    and short text correctly.
    """

    def test_five_clear_paragraphs_produce_five_chunks(self):
        """
        Five paragraphs each >= MIN_CHUNK_CHARS and <= MAX_CHUNK_CHARS
        should produce exactly 5 chunks (overlap changes size, not count).
        """
        paragraphs = []
        for i in range(1, 6):
            # ~216 chars per paragraph — safely inside [150, 1200]
            para = f"Paragraf {i}: " + (
                "Bu paragraf test içeriğidir ve yeterince uzundur. " * 4
            )
            self.assertGreaterEqual(len(para), MIN_CHUNK_CHARS,
                f"Test setup error: paragraph {i} is too short ({len(para)} chars)")
            self.assertLessEqual(len(para), MAX_CHUNK_CHARS,
                f"Test setup error: paragraph {i} is too long ({len(para)} chars)")
            paragraphs.append(para)

        text = "\n\n".join(paragraphs)
        chunks = _split_into_chunks(text)

        self.assertEqual(len(chunks), 5,
            f"Expected 5 chunks, got {len(chunks)}.\n"
            f"Chunk lengths: {[len(c) for c in chunks]}\n"
            f"Chunk previews: {[c[:60] for c in chunks]}")

    def test_1000_char_paragraph_split_into_at_least_2_chunks(self):
        """
        A 1000-character single paragraph should be split into >= 2 chunks.

        *** EXPECTED TO FAIL ***
        MAX_CHUNK_CHARS = 1200. A 1000-char paragraph is UNDER the limit
        and will NOT be split. This test deliberately reveals this design gap:
        the chunker's split threshold (1200 chars) is larger than many
        real paragraphs that contain multiple distinct ideas. A 1000-char
        block will be sent to the embedding model as a single, semantically
        diluted unit.
        """
        sentence = "Bu maaş hesaplama sürecinde dikkat edilmesi gereken kritik bir adımdır. "
        repetitions = (1000 // len(sentence)) + 1
        text = (sentence * repetitions)[:1000]
        self.assertEqual(len(text), 1000)

        chunks = _split_into_chunks(text)

        self.assertGreaterEqual(len(chunks), 2,
            f"DESIGN FINDING: A 1000-char paragraph produced only {len(chunks)} chunk. "
            f"MAX_CHUNK_CHARS={MAX_CHUNK_CHARS}. "
            f"The chunker only splits paragraphs longer than {MAX_CHUNK_CHARS} chars. "
            f"1000-char paragraphs are ingested as a single large chunk, "
            f"which dilutes embedding precision for dense regulatory text.")

    def test_short_paragraph_merged_into_next_chunk(self):
        """
        A paragraph shorter than MIN_CHUNK_CHARS should be absorbed into
        the following chunk, not stored as a tiny isolated chunk.
        """
        short_para = "Bu kısa bir paragraftır."          # ~24 chars — well under 150
        normal_para = (
            "Bu normal uzunluktaki paragraf yeterli içeriğe sahiptir ve "
            "birden fazla cümle barındırmaktadır. " * 4
        )  # ~236 chars — comfortably above 150

        self.assertLess(len(short_para), MIN_CHUNK_CHARS,
            f"Test setup: short_para must be < {MIN_CHUNK_CHARS} chars")
        self.assertGreaterEqual(len(normal_para), MIN_CHUNK_CHARS,
            f"Test setup: normal_para must be >= {MIN_CHUNK_CHARS} chars")

        text = short_para + "\n\n" + normal_para
        chunks = _split_into_chunks(text)

        self.assertEqual(len(chunks), 1,
            f"Expected short paragraph to merge into next chunk (1 chunk total), "
            f"got {len(chunks)} chunks.\n"
            f"Chunk previews: {[c[:80] for c in chunks]}")


# ===========================================================================
# TEST 3: Embedding shape
# ===========================================================================

class TestEmbeddingShape(unittest.TestCase):
    """
    Test 3: Verify the Gemini embedding API returns a valid float vector.
    Requires a live GOOGLE_API_KEY.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = _get_genai_client()
        cls.sentence = "Maaş hesaplama işlemi nasıl yapılır?"
        cls.vector = _embed_chunk(cls.client, cls.sentence)
        print(f"\n  [Test 3] Embedded: '{cls.sentence}'")
        print(f"  Embedding dimensions: {len(cls.vector)}")
        print(f"  First 5 values: {[round(v, 6) for v in cls.vector[:5]]}")

    def test_result_is_a_list(self):
        self.assertIsInstance(self.vector, list,
            f"Expected list, got {type(self.vector)}")

    def test_list_is_nonempty(self):
        self.assertGreater(len(self.vector), 0,
            "Embedding vector has zero dimensions")

    def test_all_values_are_floats(self):
        non_floats = [(i, v) for i, v in enumerate(self.vector)
                      if not isinstance(v, float)]
        self.assertEqual(len(non_floats), 0,
            f"Non-float values at positions: {non_floats[:5]}")

    def test_no_none_values(self):
        nones = [i for i, v in enumerate(self.vector) if v is None]
        self.assertEqual(len(nones), 0,
            f"None values at indices: {nones[:5]}")


# ===========================================================================
# TEST 4: ChromaDB round-trip
# ===========================================================================

class TestChromaDBRoundTrip(unittest.TestCase):
    """
    Test 4: Insert 3 test chunks into a temporary collection, query,
    verify results and metadata, then delete the collection.
    """

    TEST_COLLECTION = "test_round_trip_temp"

    @classmethod
    def setUpClass(cls):
        cls.client = _get_genai_client()
        chroma = chromadb.PersistentClient(path=CHROMA_DIR_STR)
        # Clean up any leftover from a previous failed run
        try:
            chroma.delete_collection(cls.TEST_COLLECTION)
        except Exception:
            pass
        cls.collection = chroma.get_or_create_collection(
            name=cls.TEST_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Define and embed 3 test chunks
        cls.test_chunks = [
            {
                "id": "test_chunk_0",
                "text": (
                    "Göreve başlama belgesi, personelin kuruma resmi katılışını belgeler "
                    "ve maaş ödemesi yapılabilmesi için zorunlu bir evraktır."
                ),
                "metadata": {"source": "test_explicit", "chunk_index": 0},
            },
            {
                "id": "test_chunk_1",
                "text": (
                    "İcra kesintisi, mahkeme veya icra müdürlüğünün yazılı kararına "
                    "dayanır ve net aylığın dörtte birini kesinlikle geçemez."
                ),
                "metadata": {"source": "test_explicit", "chunk_index": 1},
            },
            {
                "id": "test_chunk_2",
                "text": (
                    "Kümülatif matrah her yılın ilk bordrosunda sıfırlanmalı ve "
                    "bu işlem kayıt altına alınmalıdır; aksi halde yanlış vergi dilimi uygulanır."
                ),
                "metadata": {"source": "test_explicit", "chunk_index": 2},
            },
        ]

        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in cls.test_chunks:
            vector = _embed_chunk(cls.client, chunk["text"])
            ids.append(chunk["id"])
            embeddings.append(vector)
            documents.append(chunk["text"])
            metadatas.append(chunk["metadata"])
            time.sleep(1)  # Rate limit

        cls.collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas,
        )
        print(f"\n  [Test 4] Inserted {len(ids)} test chunks into '{cls.TEST_COLLECTION}'")

        # Run query
        cls.query_text = "göreve başlama belgesi şartları nelerdir"
        cls.query_vector = _embed_query(cls.client, cls.query_text)
        cls.query_results = cls.collection.query(
            query_embeddings=[cls.query_vector],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        cls.top_doc = cls.query_results["documents"][0][0]
        cls.top_meta = cls.query_results["metadatas"][0][0]
        cls.top_dist = float(cls.query_results["distances"][0][0])
        print(f"  Query: '{cls.query_text}'")
        print(f"  Top result: '{cls.top_doc[:80]}...'")
        print(f"  Distance:   {cls.top_dist:.4f}")
        print(f"  Metadata:   {cls.top_meta}")

    @classmethod
    def tearDownClass(cls):
        chroma = chromadb.PersistentClient(path=CHROMA_DIR_STR)
        try:
            chroma.delete_collection(cls.TEST_COLLECTION)
            print(f"\n  [Test 4] Cleaned up test collection '{cls.TEST_COLLECTION}'")
        except Exception as e:
            print(f"\n  [Test 4] Warning: could not delete test collection: {e}")

    def test_at_least_one_result_returned(self):
        self.assertGreater(len(self.query_results["documents"][0]), 0,
            "ChromaDB query returned no documents")

    def test_metadata_has_source_field(self):
        self.assertIn("source", self.top_meta,
            f"'source' key missing from metadata: {self.top_meta}")

    def test_metadata_has_chunk_index_field(self):
        self.assertIn("chunk_index", self.top_meta,
            f"'chunk_index' key missing from metadata: {self.top_meta}")

    def test_distance_is_float_between_0_and_2(self):
        self.assertIsInstance(self.top_dist, float,
            f"Distance is not a float: {type(self.top_dist)}")
        self.assertGreaterEqual(self.top_dist, 0.0,
            f"Distance is below 0: {self.top_dist}")
        self.assertLessEqual(self.top_dist, 2.0,
            f"Distance is above 2: {self.top_dist}")

    def test_top_result_is_semantically_correct(self):
        """The query about 'göreve başlama belgesi' should return chunk 0."""
        self.assertIn("göreve başlama", self.top_doc.lower(),
            f"Top result does not mention 'göreve başlama'. "
            f"Got instead: '{self.top_doc[:150]}'")


# ===========================================================================
# TEST 5: Retrieval relevance (critical — uses live tee_knowledge_base)
# ===========================================================================

class TestRetrievalRelevance(unittest.TestCase):
    """
    Test 5: Verify the retrieval system returns semantically correct chunks
    from the actual ingested tee_knowledge_base.

    Uses distance_threshold=2.0 to bypass the default 0.7 filter so we
    always get ranked results and can inspect the actual top-1 hit.
    """

    def _query(self, text: str, top_k: int = 5) -> list[dict]:
        return retrieve_context(text, top_k=top_k, distance_threshold=2.0)

    def _print_result(self, label: str, query: str, results: list[dict]) -> None:
        print(f"\n  {label} — '{query}'")
        if results:
            top = results[0]
            print(f"  Top source:   {top['source']}")
            print(f"  Top file:     {top['filename']}")
            print(f"  Distance:     {top['distance']}")
            print(f"  Text preview: {top['text'][:150]}...")
        else:
            print("  *** NO RESULTS RETURNED ***")

    def test_query_a_salary_calculation_returns_explicit(self):
        """
        'maaş hesaplama nasıl yapılır' → regulations (explicit) should rank first.
        The procedure is defined in MADDE 1-3 of the regulation, not in the interview.
        """
        query = "maaş hesaplama nasıl yapılır"
        results = self._query(query)
        self._print_result("Query A", query, results)

        self.assertGreater(len(results), 0,
            f"No results returned for query: '{query}'. "
            "Is the ChromaDB collection populated? Run ingestion first.")

        top = results[0]
        if top["source"] != "explicit":
            print(f"\n  FAILURE DETAIL — Unexpected top chunk:")
            print(f"    Source:   {top['source']}")
            print(f"    File:     {top['filename']}")
            print(f"    Distance: {top['distance']}")
            print(f"    Full text:\n    {top['text']}")

        self.assertEqual(top["source"], "explicit",
            f"Expected source='explicit', got '{top['source']}'.\n"
            f"Top chunk: '{top['text'][:300]}'\n"
            f"Distance: {top['distance']}")

    def test_query_b_new_employee_mistakes_returns_tacit(self):
        """
        'yeni başlayan personel nerede hata yapar' → interview (tacit) should rank first.
        The expert explicitly lists the top-3 mistakes in the interview transcript.
        """
        query = "yeni başlayan personel nerede hata yapar"
        results = self._query(query)
        self._print_result("Query B", query, results)

        self.assertGreater(len(results), 0,
            f"No results returned for query: '{query}'.")

        top = results[0]
        if top["source"] != "tacit":
            print(f"\n  FAILURE DETAIL — Unexpected top chunk:")
            print(f"    Source:   {top['source']}")
            print(f"    File:     {top['filename']}")
            print(f"    Distance: {top['distance']}")
            print(f"    Full text:\n    {top['text']}")

        self.assertEqual(top["source"], "tacit",
            f"Expected source='tacit', got '{top['source']}'.\n"
            f"Top chunk: '{top['text'][:300]}'\n"
            f"Distance: {top['distance']}")

    def test_query_c_icra_distance_below_1(self):
        """
        'icra kesintisi belge gereksinimi' → top result must have distance < 1.0.
        Cosine distance < 1.0 means the vectors share positive similarity,
        i.e. the retrieved chunk is genuinely on-topic, not a random match.
        """
        query = "icra kesintisi belge gereksinimi"
        results = self._query(query)
        self._print_result("Query C", query, results)

        self.assertGreater(len(results), 0,
            f"No results returned for query: '{query}'.")

        top = results[0]
        self.assertLess(top["distance"], 1.0,
            f"Top result distance {top['distance']:.4f} >= 1.0. "
            f"This means the retriever is NOT finding a genuinely relevant chunk.\n"
            f"Top chunk source: {top['source']}\n"
            f"Top chunk text: '{top['text'][:300]}'")


# ===========================================================================
# PART 3: Integration test — new document insertion
# ===========================================================================

class TestNewDocumentIntegration(unittest.TestCase):
    """
    Part 3: Full integration test.

    Inserts data/new_directive.txt into the live collection, verifies that:
    - New chunks were added
    - Retrieval correctly surfaces the new document for related queries
    - Existing documents were NOT wiped
    - generate_process_map() still produces valid JSON
    """

    NEW_DOC_PATH = str(BASE_DIR / "data" / "new_directive.txt")

    @classmethod
    def setUpClass(cls):
        """Run the insertion once; all test methods use the results."""
        chroma = chromadb.PersistentClient(path=CHROMA_DIR_STR)
        collection = chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        cls.count_before = collection.count()
        print(f"\n  [Part 3] Chunk count BEFORE insertion: {cls.count_before}")

        cls.summary = insert_new_document(cls.NEW_DOC_PATH, "explicit")
        print(f"  New chunks added:    {cls.summary['new_chunks']}")
        print(f"  Collection size now: {cls.summary['collection_size']}")

        cls.collection = collection

    def test_new_chunks_were_added(self):
        self.assertGreater(self.summary["new_chunks"], 0,
            "insert_new_document() returned 0 new chunks. "
            "Check that data/new_directive.txt exists and is non-empty.")

    def test_collection_size_increased_correctly(self):
        expected = self.count_before + self.summary["new_chunks"]
        actual = self.summary["collection_size"]
        self.assertEqual(actual, expected,
            f"Expected collection size {expected}, got {actual}. "
            f"Possible ID collision (upsert may have overwritten existing chunks).")

    def test_new_document_is_top_result_for_directive_query(self):
        """After insertion, querying for Temmuz katsayı should surface new_directive.txt."""
        query = "Temmuz ayı maaş katsayı değişikliği"
        results = retrieve_context(query, top_k=5, distance_threshold=2.0)

        print(f"\n  Query: '{query}'")
        if results:
            top = results[0]
            print(f"  Top file:     {top['filename']}")
            print(f"  Top source:   {top['source']}")
            print(f"  Distance:     {top['distance']}")
            print(f"  Text preview: {top['text'][:200]}...")
        else:
            print("  *** NO RESULTS ***")

        self.assertGreater(len(results), 0,
            f"No results for query '{query}'")

        top = results[0]
        self.assertEqual(top["filename"], "new_directive.txt",
            f"Expected 'new_directive.txt' as top result, got '{top['filename']}'.\n"
            f"Text: '{top['text'][:200]}'\n"
            f"Distance: {top['distance']}")

    def test_existing_mevzuat_chunks_still_present(self):
        """The regulation document chunks must still be in the collection."""
        existing = self.collection.get(where={"filename": "explicit_mevzuat.txt"})
        self.assertGreater(len(existing["ids"]), 0,
            "explicit_mevzuat.txt chunks are GONE — the collection was wiped! "
            "insert_new_document() must never drop existing data.")

    def test_existing_tacit_interview_chunks_still_present(self):
        """The anonymised interview chunks must still be in the collection."""
        tacit = self.collection.get(where={"source": "tacit"})
        self.assertGreater(len(tacit["ids"]), 0,
            "Tacit interview chunks are GONE — the collection was wiped! "
            "insert_new_document() must never drop existing data.")

    def test_generate_process_map_returns_valid_json_after_insertion(self):
        """
        After new-document insertion, generate_process_map() must return
        a valid JSON dict with a non-empty 'steps' list.
        """
        result = generate_process_map()

        print("\n  === FULL PROCESS MAP OUTPUT ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("  ================================")

        # Validity checks
        self.assertIsInstance(result, dict,
            f"generate_process_map() returned {type(result)}, not dict")

        self.assertNotIn("hata", result,
            f"generate_process_map() returned an error:\n"
            f"  hata: {result.get('hata')}\n"
            f"  ham_çıktı: {str(result.get('ham_çıktı', ''))[:300]}")

        # Round-trip JSON validity
        try:
            json_str = json.dumps(result, ensure_ascii=False)
            parsed_back = json.loads(json_str)
            self.assertEqual(result, parsed_back)
        except (TypeError, json.JSONDecodeError) as e:
            self.fail(f"Output cannot be round-tripped through JSON: {e}")

        self.assertIn("steps", result,
            f"'steps' key missing. Keys present: {list(result.keys())}")

        if "steps" in result:
            self.assertIsInstance(result["steps"], list)
            self.assertGreater(len(result["steps"]), 0,
                "'steps' list is empty")

        # Informational: does the directive content appear in the output?
        if "steps" in result:
            output_str = json.dumps(result, ensure_ascii=False).lower()
            directive_keywords = ["temmuz", "katsayı", "0.05", "genelge", "artırıl"]
            found = [kw for kw in directive_keywords if kw in output_str]
            print(f"\n  Directive-specific keywords found in process map: {found or 'NONE'}")
            if found:
                print("  → New directive content IS reflected in the process map.")
            else:
                print(
                    "  → New directive content is NOT visible in the process map.\n"
                    "  (Expected: the process map query 'maaş hesaplama adımları' "
                    "may not match the directive's coefficient-change topic closely "
                    "enough to retrieve it in the top-8 context chunks.)"
                )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
