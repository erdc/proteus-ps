def load_macros(filename,verbose=False):
    from IPython.display import Markdown
    with open(filename,'r') as macros:
        markdown = ""
        for line in macros.readlines():
            if len(line.strip('\n')) and not line.startswith('%'):
                markdown += "${0}$\n".format(line.strip('\n'))
        markdown+=('Loaded macros\n')
        if verbose:
            print(markdown)
        return Markdown(markdown)
