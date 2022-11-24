import time
from typing import cast
from yapydi import bean, injected, BeanInitializer, Injected, Scope, logger, BeanFactory, enable_di

#######################Testing Area########################
def setup_module(module):
    BeanInitializer().initialize()
    logger.info(f"do setup for {module}")


def teardown_module(module):
    logger.info(f"do tearown for {module}")


BEAN_TEST_STR_CONTENT = "that is my test"


class MyItem:
    def __init__(self, name: str):
        self.name = name
        logger.debug("Init MyItem")

    def __str__(self):
        return f"MyItem(name={self.name})"


class MyInjection:
    @injected
    def __init__(self, test: str = Injected[str, "bean_test_str"]):
        logger.info(f"{self.__class__} __init__ self={self}, got bean_test_str={test}")
        self.bean_test_str = test


@bean()
class MyConfigturation:
    hard_to_tell_content = "It's hard to tell night time from the day."

    @injected
    def __init__(self, test: str = Injected[str, "bean_test_str"]):
        logger.info(f"{self.__class__} __init__ self={self}, got bean_test_str={test}")
        self.bean_test_str = test

    @bean()
    def hard_to_tell(self) -> str:
        return self.hard_to_tell_content

    @bean()
    @injected
    def internal_injected_bean(self, test: str = Injected[str, "bean_test_str"]) -> str:
        return test

    @injected
    @bean()
    def internal_injected_bean_2(self, test: str = Injected[str, "bean_test_str"]) -> str:
        return test


@bean()
def bean_test_str() -> str:
    return BEAN_TEST_STR_CONTENT


@bean()
@injected
def my_item_factory(test: str = Injected[str, "bean_test_str"]) -> MyItem:
    logger.debug("Real method is invoking")
    result = MyItem(test)
    logger.debug(f"MyItem created {result}")
    return result


@injected
@bean(scope=Scope.PROTOTYPE)
def another_bean(test: str = Injected[str, "bean_test_str"]) -> str:

    return f"raw_str plus {test}: {time.time()}"


@injected
def my_test_method(
    test: str = Injected[str, "bean_test_str"],
    another_str: str = Injected[str, "another_bean"],
    my_item: MyItem = Injected[MyItem, "my_item_factory"],
):
    logger.info(f"Invoking my_test_method, test_str='{test}', another_bean='{another_str}', item={my_item}")
    return test


@injected
def another_method(test: str = Injected[str, "another_bean"]):
    return test


@injected
def injected_by_name(another_bean: str = ""):  # pylint: disable=redefined-outer-name
    return another_bean


def general_smoke():
    my_test_method()
    another_method()
    registry = BeanFactory.registry()
    item = registry.one_by_type(MyItem)
    assert item is not None
    injected_by_name()


def test_basic_inject():
    assert my_test_method() == BEAN_TEST_STR_CONTENT
    assert another_bean().count(BEAN_TEST_STR_CONTENT) > 0
    # prototype bean need to be new instance every time
    assert another_bean() != another_bean()
    item = my_item_factory()
    assert item is not None
    assert item.name == BEAN_TEST_STR_CONTENT


def test_bean_retrieve() -> None:
    registry = BeanFactory.registry()
    assert registry is not None
    item = registry.one_by_name("my_item_factory")
    assert item is not None
    assert isinstance(item, MyItem)
    assert item.name == BEAN_TEST_STR_CONTENT

    my_item: MyItem = cast(MyItem, registry.one_by_type(MyItem))
    assert my_item is not None
    assert isinstance(my_item, MyItem)
    assert my_item.name == BEAN_TEST_STR_CONTENT


def test_inject_without_default_value():
    value = injected_by_name()
    assert value is not None
    assert value.count(BEAN_TEST_STR_CONTENT) > 0


def test_injection_inside_class():
    item = MyInjection()
    content = item.bean_test_str
    assert content is not None
    logger.info(f"item.another_bean = {content}")
    assert content == BEAN_TEST_STR_CONTENT


def test_injection_by_configuration() -> None:
    registry = BeanFactory.registry()
    hard_to_tell_content = MyConfigturation.hard_to_tell_content
    test_str = BEAN_TEST_STR_CONTENT
    item: MyConfigturation = cast(MyConfigturation, registry.one_by_type(MyConfigturation))
    assert item is not None
    assert item.bean_test_str is not None
    assert item.bean_test_str == test_str
    assert hard_to_tell_content == registry.one_by_name("hard_to_tell")
    assert test_str == registry.one_by_name("internal_injected_bean")
    assert test_str == registry.one_by_name("internal_injected_bean_2")


@enable_di
def main():
    logger.info("Hello my injection")
    test_injection_by_configuration()


if __name__ == "__main__":
    main()
