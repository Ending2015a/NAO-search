from logging.config import dictConfig

class LoggingConfig:
    @staticmethod
    def Use(filename='training.log', level='INFO'):
        dictConfig({
            'version': 1,
            'formatters':{
                'default': {
                    'format': '[%(asctime)s|%(threadName)s|%(levelname)s:%(module)s]: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                    }
                },
            'handlers': {
                'stream': {
                    'level': level,
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout',
                    'formatter': 'default'
                    },
                'file': {
                    'level': level,
                    'filename': filename,
                    'class': 'logging.FileHandler',
                    'formatter': 'default',
                    }
                },
            'root': {
                'level': level,
                'handlers': ['stream', 'file']
                }
            }
        )

