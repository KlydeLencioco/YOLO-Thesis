import math
from copy import deepcopy


class CentroidTracker:
    def __init__(self, max_life: int = 30, max_distance: int = 60):
        self.objects = {}
        self.object_id = 0

        self.max_life = max_life
        self.life_counter = {}

        self.max_distance = max_distance

        self.deleted = None

    def add(self, centroid):
        self.object_id += 1
        self.objects[self.object_id] = [centroid]
        self.life_counter[self.object_id] = 0

    def delete(self, object_id):
        self.objects.pop(object_id)
        self.life_counter.pop(object_id)
        self.deleted.append(object_id)

    def update(self, object_id, centroid):
        if self.objects[object_id][-1] != centroid:
            self.objects[object_id].append(centroid)
        self.life_counter[object_id] = 0

    def track(self, new_centroids):
        self.deleted = []

        for object_id, life_count in self.life_counter.items():
            self.life_counter[object_id] = life_count + 1

        if not self.objects:
            for centroid in new_centroids:
                self.add(centroid)

            return self.objects

        distances = []
        for object_id, current_centroid in self.objects.items():
            latest = current_centroid[-1]
            for index, new_centroid in enumerate(new_centroids):
                # dist = math.dist(latest, new_centroid)
                dist = math.sqrt(
                    sum([(a - b) ** 2 for a, b in zip(latest, new_centroid)])
                )
                distances.append((object_id, new_centroid, dist))

        distances_asc = sorted(distances, key=lambda x: x[2])

        tracked_objects = []
        tracked_centroids = []
        new_limit = len(new_centroids)
        skipped = []

        for object_id, new_centroid, dist in distances_asc:
            if new_centroid in tracked_centroids:
                continue

            if dist > self.max_distance:
                continue

            if object_id in tracked_objects:
                skipped.append((object_id, new_centroid))
            else:
                self.update(object_id, new_centroid)

                tracked_objects.append(object_id)
                tracked_centroids.append(new_centroid)

        for object_id, new_centroid in skipped:
            if len(tracked_objects) == new_limit:
                break

            if object_id in tracked_objects:
                self.add(new_centroid)
                tracked_objects.append(self.object_id)
                tracked_centroids.append(new_centroid)

        life_counter = deepcopy(self.life_counter)
        for object_id, life_count in life_counter.items():
            if life_count > self.max_life:
                self.delete(object_id)

        return self.objects
