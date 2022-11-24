from yapydi import enable_di, bean, BeanRegistry, Injected, injected


@bean(name="my_bean_name")
def world() -> str:
    return "World"


def hello_world():
    registry = BeanRegistry.get_instance()
    str_bean = registry.one_by_name("my_bean_name")
    print(f"Hello {str_bean}!")
    assert str_bean == "World"


@injected
def test_injection(injected_world: str = Injected[str, "my_bean_name"]):
    print(f"Hello injected {injected_world}!")
    assert injected_world == "World"


@enable_di
def main():
    hello_world()
    test_injection()


if __name__ == "__main__":
    main()
