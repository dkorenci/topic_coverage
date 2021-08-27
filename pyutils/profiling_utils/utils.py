import cProfile

def profileCallable(callable):
    profile = cProfile.Profile()
    try:
        profile.enable()
        callable()
        profile.disable()
    finally:
        profile.print_stats()