import os
from pathlib import Path

from rag import RAG


PROJECT_ROOT = Path(
    os.environ["UNIVERSAL_RAG_PROJECT_ROOT"]
)


def main():
    rag = RAG(
        project_root=PROJECT_ROOT,
        candidate_limit=30,
        result_limit=8,
        max_context_chars=16000,
    )

    try:
        question = input(
            f"\nQuestion about {PROJECT_ROOT.name}: "
        ).strip()

        if not question:
            print("No question provided.")
            return

        result = rag.ask(question)

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(result["answer"])

        print("\n" + "=" * 70)
        print("RETRIEVED CHUNKS")
        print("=" * 70)

        for index, item in enumerate(
            result["results"],
            start=1,
        ):
            payload = item.get(
                "payload",
                {},
            )

            print(
                f"\n[{index}] "
                f"score={item.get('score')}"
            )

            print(
                f"File: "
                f"{payload.get('source')}"
            )

            print(
                f"Chunk: "
                f"{payload.get('chunk_index')}"
            )

            print("-" * 50)

            text = payload.get(
                "text",
                "",
            )

            print(text[:1500])

        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)

        for source in result["sources"]:
            print(source["source"])

    finally:
        rag.close()


if __name__ == "__main__":
    main()
