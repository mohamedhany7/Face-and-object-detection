class Person:
    def __init__(self,image, name, bbox = None, isTracked = False, traker = None):
        self.image = image
        self.name = name
        self.bbox = bbox
        self.isTracked = isTracked
        self.tracker = traker
    
    def get_image(self):
        return self.image
    
    def get_name(self):
        return self.name
    
    def set_bbox(self,bbox):
        self.bbox = bbox

    def get_bbox(self):
        return self.bbox

    def updateTracker(self,img):
        success, bbox = self.tracker.update(img)
        if success:
            self.image = img
            self.bbox = bbox
        return success

    def set_isTracked(self):
        self.isTracked = True

    def get_isTracked(self):
        return self.isTracked

    def lost(self):
        self.isTracked = False

    #def set_faceLoc(self,faceLoc):
    #    self.faceloc = faceLoc

    #def get_faceLoc(self):
    #    return self.faceLoc