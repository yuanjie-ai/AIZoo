#!/usr/bin/env bash

git init
git add * # add vsc
git commit -m "init"

git remote add origin git@github.com:Jie-Yuan/aizoo.git
git branch -M master
git push -u origin master
# git remote remove origin
