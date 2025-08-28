"""Test file for outline view functionality."""

class TestClass:
    """A simple test class."""
    
    def __init__(self):
        """Initialize the test class."""
        self.value = 42
    
    def test_method(self):
        """A test method."""
        return "Hello, World!"
    
    def another_method(self, param: str) -> str:
        """Another test method with type hints."""
        return f"Parameter: {param}"


def test_function():
    """A standalone test function."""
    return "This is a function"


def main():
    """Main function."""
    test_obj = TestClass()
    print(test_obj.test_method())
    print(test_obj.another_method("test"))
    print(test_function())


if __name__ == "__main__":
    main()
