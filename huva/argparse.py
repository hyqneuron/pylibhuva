import argparse

def clear_arg_name(raw_name):
    while len(raw_name) > 0 and raw_name[0]=='-':
        raw_name = raw_name[1:]
    return raw_name.replace('-', '_')


class RootedArgumentParser(argparse.ArgumentParser):
    """
    Argument parser with one hierarchy
    arguments may be specified with a root, and can be accessed at args.root_name.arg_name
    e.g.
    parser = RootedArgumentParser()
    parser.add_argument('--cat-age', type=int, root='animals')
    args = parser.parse_args()
    cat_age = args.animals.cat_age

    """

    def __init__(self):
        super(ArgumentParser, self).__init__()
        self.roots = {} # maps root_name to [arg_name]

    def add_argument(self, *args, **kwargs):
        root_name = None
        if 'root' in kwargs:
            root_name = kwargs['root']
            del kwargs['root']
        if len(args)==1:
            arg_name = clear_arg_name(args[0])
        elif len(args==2):
            arg_name = clear_arg_name(args[1])
        else:
            raise RuntimeError, 'not sure how to handle more than 2 positional arguments for {}'.format(args)
        if root_name not in self.roots:
            self.roots[root_name] = []
        if arg_name not in self.roots[root_name]:
            self.roots[root_name].append(arg_name)
        super(ArgumentParser, self).add_argument(*args, **kwargs)

    def parse_args(self):
        args = super(ArgumentParser, self).parse_argument()
        # create one sub-namespace per root
        for root_name in self.roots:
            args.__dict__[root_name] = argparse.Namespace()
        # if arg_name is in root's namelist, put the value under corresponding root's namespace
        for arg_name, arg_value in args.__dict__.iteritems():
            relevant_rnames = [root_name for root_name in self.roots if arg_name in self.roots[root_name]]
            for root_name in relevant_rnames:
                args.__dict__[root_name].__dict__[arg_name] = arg_value
        return args

