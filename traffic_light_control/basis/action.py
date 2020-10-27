
class Action(object):
    def __init__(self, intersection_id, phase_id):
        self.intersection_id = intersection_id
        self.phase_id = phase_id

    def get_phase_id(self):
        return self.phase_id

    def get_intersection_id(self):
        return self.intersection_id
