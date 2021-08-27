'''
Setup logging for bigartm, this module should be imported before other artm
code because otherwise the other import will initialize logging.
'''

__loggingSetup=False

def __setUpLogging():
    # https://bigartm.readthedocs.io/en/stable/api_references/c_interface.html#artmconfigurelogging
    global __loggingSetup
    import artm
    lc = artm.messages.ConfigureLoggingArgs()
    # configure logging folder
    lc.log_dir = '.'
    # effectively disable logging via loglevel 3
    lc.minloglevel = 3  # 0 = INFO, 1 = WARNING, 2 = ERROR, 3 = FATAL
    lc.stop_logging_if_full_disk = True
    lc.max_log_size = 1  # in MB
    artm.wrapper.LibArtm(logging_config=lc)
    __loggingSetup = True

if not __loggingSetup:
    __setUpLogging()