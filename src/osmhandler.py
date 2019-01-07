import osmium
import os
import pickle
import numpy as np


class Handler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.facilities = []
        self.tags = {
            'shop': ('mall', 'department_store', 'supermarket'),
             'office': ('company', 'financial', 'therapy'),
            #'leisure': ('all',),
            #'tourism': ('all',),
            'amenity': ('hospital','clinic','pharmacy','theatre','marketplace','postbox')
                }

    def get_facilities(self, n):
        for tag in self.tags:
            subtags = self.tags[tag]
            if tag in n.tags:
                if 'all' in subtags:
                    self.facilities.append((n.location.lat, n.location.lon))
                else:
                    for subtag in subtags:
                        if subtag in n.tags[tag]:
                            self.facilities.append(
                                    (n.location.lat, n.location.lon)
                                    )

    def node(self, n):
        self.get_facilities(n)

if __name__=='__main__':
    h = Handler()
    h.apply_file('../osm_maps/Moscow.osm.pbf')
    with open('locations.pickle', 'wb') as file:
        pickle.dump(np.array(h.facilities), file)

    print('done')
