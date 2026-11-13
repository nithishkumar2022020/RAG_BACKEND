from unittest.mock import patch

from app.core.config import Settings
from app.services.rag import SYSTEM_PROMPT, RagService


class FakeStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.document_queries = []

    def search(self, vector, limit, document_id=None, document_ids=None):
        return self.chunks[:limit]

    def all_chunks(self, document_id=None, document_ids=None):
        return self.chunks

    def get_all_chunks(self, document_id):
        self.document_queries.append(document_id)
        return self.chunks


def test_answer_prompts_for_general_and_judgment_questions(capsys):
    fake_sources = [
        {
            "id": "chunk-1",
            "document_id": "resume-1",
            "filename": "resume.pdf",
            "page": 1,
            "chunk_index": 0,
            "text": (
                "The candidate is an AI engineer with experience in Python and PyTorch. "
                "They built an image classifier that reached 94% accuracy. "
                "Contact details are not included in this excerpt."
            ),
        },
        {
            "id": "chunk-2",
            "document_id": "resume-1",
            "filename": "resume.pdf",
            "page": 2,
            "chunk_index": 1,
            "text": "The candidate deployed an ML service used by three internal teams.",
        },
    ]
    questions = [
        "What is this about?",
        "Score this resume 1-10 for an AI/ML engineer role. Consider projects, technical skills, measurable impact, and clarity.",
        "You be the judge.",
        "What is the candidate's phone number?",
    ]
    store = FakeStore(fake_sources)
    settings = Settings(
        gemini_api_key="test-key",
        chunking_strategy="fixed",
        embedding_requests_per_minute=99,
        _env_file=None,
    )

    with patch("app.services.rag.genai.Client") as client_factory:
        service = RagService(settings, store)
        service._embed = lambda texts, task: [[1.0, 0.0] for _ in texts]

        def generate_content(*, model, contents, config):
            assert config.system_instruction == SYSTEM_PROMPT
            assert contents.startswith("Question: ")
            assert "\n\nExcerpts (JSON):\n" in contents
            if "phone number" in contents:
                answer = "The phone number is not present in the excerpts."
            elif "What is this about?" in contents:
                answer = "This resume describes an AI engineer's experience [resume.pdf, p. 1]."
            elif "Score this resume" in contents:
                answer = "I rate it 8/10 based on its projects and skills [resume.pdf, p. 1]."
            else:
                answer = "I judge the resume as promising based on its project impact [resume.pdf, p. 2]."
            print(answer)
            return type("Response", (), {"text": answer})()

        service.client.models.generate_content.side_effect = generate_content
        outputs = [
            service.answer(question, document_id="resume-1") for question in questions
        ]

    printed = capsys.readouterr().out
    assert printed.count("\n") == 4
    assert all(output["answer"] for output in outputs)
    assert all(output["citations"] for output in outputs)
    assert store.document_queries == ["resume-1"] * 3
    calls = client_factory.return_value.models.generate_content.call_args_list
    assert all("filename" in call.kwargs["contents"] for call in calls)
    assert "phone number is not present" in outputs[3]["answer"]
