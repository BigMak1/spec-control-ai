def pytest_addoption(parser):
    parser.addoption("--run-eval", action="store_true", default=False, help="Run eval tests")
