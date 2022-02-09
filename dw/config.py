from yaml import load, Loader
import os

class Config(object):

    def __init__(self, path=None, yaml=None, root=""):
        """ Generate config object.
        Either path or yaml dict has to be provided.
        All configurations can be accessed in object style.
        BASE config will only be considered if the path is given.

        :param path: path to a config file
        :param yaml: yaml data as dict
        :param root: root of the config (will be automatically identified if path is given and used to replace ~CONFIG)
        """

        assert (path is None) ^ (yaml is None), "Either Path or YAML dict has to be provided"
        if not path is None:
            with open(path, 'r') as file:
                self.absfile = os.path.abspath(path)
                self.root = os.sep.join(self.absfile.split(os.sep)[:-1])
                self.raw = file.read()
                self.raw = self.raw.replace("~CONFIG", self.root)
                self.yaml = load(self.raw, Loader)
        else:
            self.yaml = yaml
            self.raw = ""
            self.root = root

        if 'BASE' in self.yaml.keys() and not path is None:
            print(self.yaml['BASE'])
            base = Config(path=self.yaml['BASE'])
            self.join_base(base)
        else:
            self.set_attributes()

    def set_attributes(self):
        """ Transform yaml to attributes
        """

        for key, v in self.yaml.items():
            if key == 'BASE': continue
            if isinstance(v, (list, tuple)):
                setattr(self, key, [Config(yaml=x, root=self.root) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, key, Config(yaml=v, root=self.root) if isinstance(v, dict) else v)

    def join_base(self, base):
        """ Update this config with base yaml file.
        Shared arguments will be overwritten by self

        :param base: base config object to be joined
        """
        self.raw = f'#INHERITED FROM {base.absfile}\n{base.raw}\n#END INHERITANCE\n\n{self.raw}'
        self.yaml = load(self.raw, Loader)
        self.set_attributes()

    def __str__(self):
        return self.raw