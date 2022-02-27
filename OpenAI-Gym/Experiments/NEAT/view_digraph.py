#!/usr/bin/env python

from graphviz import Source

s = Source.from_file('Digraph.gv')
s.view()