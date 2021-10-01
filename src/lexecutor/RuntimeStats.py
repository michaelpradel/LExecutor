class RuntimeStats:
    total_uses = 0
    guided_uses = 0

    covered_iids = set()

    def cover_iid(self, iid):
        self.covered_iids.add(iid)

    def print(self):
        print(f"Covered iids: {len(self.covered_iids)}")
        print(f"Guided uses : {self.guided_uses}/{self.total_uses}")
