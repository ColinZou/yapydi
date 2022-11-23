from yapydi import bean, injected, BeanInitializer, Injected, Scope, logger, BeanFactory, enable_di
from typing import cast


#######################Testing Area########################
def setup_module(module):
    BeanInitializer().initialize()
    logger.info("do setup for {}".format(module))


def teardown_module(module):
    logger.info("do tearown for {}".format(module))


bean_test_str_content = "that is my test"


class MyItem:
    def __init__(self, name: str):
        self.name = name
        logger.debug("Init MyItem")

    def __str__(self):
        return "MyItem(name={})".format(self.name)


class MyInjection:
    @injected
    def __init__(self, test: str = Injected[str]):
        logger.info("{} __init__ self={}, got bean_test_str={}".format(self.__class__, self, bean_test_str))
        self.bean_test_str = test


@bean()
class MyConfigturation:
    hard_to_tell_content = "It's hard to tell night time from the day."

    @injected
    def __init__(self, bean_test_str: str = Injected[str]):
        logger.info("{} __init__ self={}, got bean_test_str={}".format(self.__class__, self, bean_test_str))
        self.bean_test_str = bean_test_str

    @bean()
    def hard_to_tell(self) -> str:
        return self.hard_to_tell_content

    @bean()
    @injected
    def internal_injected_bean(self, bean_test_str: str = Injected[str]) -> str:
        return bean_test_str

    @injected
    @bean()
    def internal_injected_bean_2(self, bean_test_str: str = Injected[str]) -> str:
        return bean_test_str


@bean()
def bean_test_str() -> str:
    return bean_test_str_content


@bean()
@injected
def my_item_factory(bean_test_str: str = Injected[str]) -> MyItem:
    logger.debug("Real method is invoking")
    result = MyItem(bean_test_str)
    logger.debug("MyItem created {}".format(result))
    return result


@injected
@bean(scope=Scope.PROTOTYPE)
def another_bean(bean_test_str: str = Injected[str]) -> str:
    import time

    return "raw_str plus {}: {}".format(bean_test_str, time.time())


@injected
def my_test_method(
    bean_test_str: str = Injected[str], another_bean: str = Injected[str], my_item_factory: MyItem = Injected[MyItem]
):
    logger.info(
        "Invoking my_test_method, test_str='{}', another_bean='{}', item={}".format(
            bean_test_str, another_bean, my_item_factory
        )
    )
    return bean_test_str


@injected
def another_method(test: str = Injected[str, "another_bean"]):
    return test


@injected
def injected_by_name(another_bean: str = ""):
    return another_bean


def general_smoke():
    my_test_method()
    another_method()
    registry = BeanFactory.registry()
    item = registry.one_by_type(MyItem)
    assert item is not None
    injected_by_name()


def test_basic_inject():
    assert my_test_method() == bean_test_str_content
    assert another_bean().count(bean_test_str_content) > 0
    # prototype bean need to be new instance every time
    assert another_bean() != another_bean()
    item = my_item_factory()
    assert item is not None
    assert item.name == bean_test_str_content


def test_bean_retrieve() -> None:
    registry = BeanFactory.registry()
    assert registry is not None
    item = registry.one_by_name("my_item_factory")
    assert item is not None
    assert type(item) == MyItem
    assert item.name == bean_test_str_content

    my_item: MyItem = cast(MyItem, registry.one_by_type(MyItem))
    assert my_item is not None
    assert type(my_item) == MyItem
    assert my_item.name == bean_test_str_content


def test_inject_without_default_value():
    value = injected_by_name()
    assert value != None
    assert value.count(bean_test_str_content) > 0


def test_injection_inside_class():
    item = MyInjection()
    content = item.bean_test_str
    assert content is not None
    logger.info("item.another_bean = {}".format(content))
    assert content == bean_test_str_content


def test_injection_by_configuration() -> None:
    registry = BeanFactory.registry()
    hard_to_tell_content = MyConfigturation.hard_to_tell_content
    test_str = bean_test_str_content
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
