#!/usr/bin/env python2
# coding=utf-8
import pkgutil
import discodop

def load_all(directory):
    for loader, name, ispkg in pkgutil.walk_packages([directory]):
        module = loader.find_module(name).load_module(name)
        exec('%s = module' % name)

if __name__ == '__main__':
	print 'Discodop test!!'

	load_all("discodop")

	dir(discodop)