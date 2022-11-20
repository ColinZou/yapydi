# YAPYDI: Yet Another Python Dependency Injection

Why? Just need a simpler DI, checked some existing ones, not a fun, so developed this one.

Why DI in python? Well, it is NOT for everyone. It is always an good idea for loose coupling for
user and implmentation. 

## How to use it
### Install
```bash
pip install yapydi
```

### Basic usage
```python
from yapydi import enable_di, bean, injected, Injected

@bean()
def world() -> str:
    return "World"
    
@injected
def hello_world(world : str = Injected[str]):
    print("Hello {}!".format(world))

@enable_di
def main():
    hello_world()

if __name__ == "__main__":
    main()

```

### Advanced usage


```python
from yapydi import enable_di, bean, BeanRegistry

@bean(name="my_bean_name")
def world() -> str:
    return "World"

def hello_world():
    registry = BeanRegistry.get_instance()
    print("Hello {}!".format(registry.one_by_name("my_bean_name")))

@enable_di
def main():
    hello_world()

if __name__ == "__main__":
    main()
```

Check more details inside * injection_test.py * please.

## How does it work
Doing bean definition and injection with annotation power. @bean() annotation will get the bean to be registered. 
@enable_di will init the beans, and @injected will be hooked so the injection can happen. BeanRegistry is the only place
for retrieve beans by name type or both.



