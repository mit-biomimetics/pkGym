class PerIterationLogs:
    def __init__(self):
        self.targets = {}
        self.logs = {}
        return None

    def register_items(self, category, target, attribute_list):
        if category in self.targets.keys():
            assert (
                self.targets[category] == target
            ), "Category already registered with different target"
        else:
            self.targets[category] = target
            self.logs[category] = {}

        for attribute in attribute_list:
            self.logs[category][attribute] = 0.0

    def log(self, category):
        for key in self.logs[category].keys():
            self.logs[category][key] = getattr(self.targets[category], key)

    def get_all_logs(self, category):
        return self.logs[category]

    def get_logs(self, category, key_list):
        return {key: self.logs[category][key] for key in key_list if key in self.logs}
