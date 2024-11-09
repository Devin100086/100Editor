import copy

from lumina3D_utils.dict_utils import equal_dicts


class RendererWrapper:
    def __init__(self, renderer, update_all_the_time):
        self.renderer = renderer
        self._cur_args = None
        self.result = None
        self.update_all_the_time = update_all_the_time

    def set_args(self, type, **args):
        something_changed = not equal_dicts(args, self._cur_args)
        if something_changed or self.update_all_the_time[type]:
            self.result = self.renderer[type].render(**args)
            self._cur_args = copy.deepcopy(args)
    
    def close(self):
        for render in self.renderer.values():
            render.close()

