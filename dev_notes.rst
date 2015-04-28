workon on new feature
---------------------
1) private-fastFM-core/new_feature

release new feature
-------------------
2) git fetch upstream
3) git checkout master
4) git merge upstream/master
5) git merge new_feature
5.1) git push origin

change to public repo ``fastFM-core``
------------------------------------
6) git fetch private
7) git branch release
8) git checkout release
9) git merge private/master
10) git checkout master
11) git merge --squash release
12) git commit -m 'new feature'
13) git push origin
14) clean up: git branch -D release
