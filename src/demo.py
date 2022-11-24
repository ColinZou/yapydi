from yapydi import enable_di, bean, BeanRegistry


@bean(name="my_bean_name")
def world() -> str:
    return "World"


def hello_world():
    registry = BeanRegistry.get_instance()
    str_bean = registry.one_by_name("my_bean_name")
    print(f"Hello {str_bean}!")


@enable_di
def main():
    hello_world()


if __name__ == "__main__":
    main()
