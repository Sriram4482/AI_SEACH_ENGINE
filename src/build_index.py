# src/build_index.py
from .search_engine import SearchEngine


def main():
    engine = SearchEngine()
    engine.build_index()


if __name__ == "__main__":
    main()
