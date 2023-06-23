from git import Repo
import re

# Helper script to find commits that are likely single-function refactorings.
#
# Dump output into a file "out" and then open the commit links in a browser with:
# for l in `cat out | xargs`; do firefox $l; done


# url_prefix = "https://github.com/scrapy/scrapy/commit/"
# repo = Repo("data/repos/scrapy")

# url_prefix = "https://github.com/nvbn/thefuck/commit/"
# repo = Repo("data/repos/thefuck")

# url_prefix = "https://github.com/scikit-learn/scikit-learn/commit/"
# repo = Repo("data/repos/scikit-learn")

url_prefix = "https://github.com/psf/black/commit/"
repo = Repo("data/repos/black")

# url_prefix = "https://github.com/pallets/flask/commit/"
# repo = Repo("data/repos/flask")

# url_prefix = "https://github.com/pandas-dev/pandas/commit/"
# repo = Repo("data/repos/pandas")

commits = list(repo.iter_commits("main"))
nb_commits_match = 0
for c in commits:
    if len(c.parents) == 0:
        continue
    diff = c.parents[0].diff(c, create_patch=True)
    if len(diff) == 1 and diff[0].a_path and diff[0].a_path.endswith(".py"):
        diff_str = str(diff[0])
        matches = re.findall(r"@@", diff_str)
        if len(matches) == 2:
            # print(diff_str)
            # print("\n---------------------------\n")
            print(f"{url_prefix}{c.hexsha}")
            nb_commits_match += 1
print(f"{len(commits)} total commits, {nb_commits_match} matches")
