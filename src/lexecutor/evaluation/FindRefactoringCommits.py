from git import Repo

# Helper script to find commits that are likely single-function refactorings.
#
# Dump output into a file "out" and then open the commit links in a browser with:
# for l in `cat out | xargs`; do firefox $l; done


# url_prefix = "https://github.com/scrapy/scrapy/commit/"
# repo = Repo("data/repos/scrapy")

# url_prefix = "https://github.com/nvbn/thefuck/commit/"
# repo = Repo("data/repos/thefuck")

url_prefix = "https://github.com/scikit-learn/scikit-learn/commit/"
repo = Repo("data/repos/scikit-learn")

commits = list(repo.iter_commits("main"))
for c in commits:
    if "refactor" in c.message:
        diff = c.parents[0].diff(c, create_patch=True)
        if len(diff) == 1 and diff[0].a_path.endswith(".py"):
            diff_str = str(diff[0])
            # heuristic check for single-function edits
            if diff_str.count("def ") <= 1:
                print(f"{url_prefix}{c.hexsha}")
