
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/ryan/Documents/PASCAL_VOC2012'  # folder that contains VOCdevkit/.

        elif database == 'sbd':
            return '/home/ryan/Documents/SBD'  # folder with img/, inst/, cls/, etc.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models/'
