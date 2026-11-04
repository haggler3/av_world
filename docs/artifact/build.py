"""Builds docs/artifact/index.html (the published page) from the numbers in results/. Run from the repo root: python docs/artifact/build.py"""
import json

H = open("docs/artifact/template.html").read()
open("docs/artifact/index.html", "w").write(H)
print(len(H))
