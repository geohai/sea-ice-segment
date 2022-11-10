# unused that might be useful at some point

    def get_full_overlapping(self, idx, bbox)->dict:
        """
        Returns the full overllaping dictionary self.samples_dict_full inside the 
        bbox

        Args:
            idx (int): rasters index in the provided df and self.dict_xar)
            bbox (iterable): minx, maxx, miny, maxy bounding box.
        """
        
        # TODO: something better than looping?
        self.samples_dict_full = {}

        # first compute the number of steps for np.linspace
        minx, maxx, miny, maxy = bbox
        n_steps_x = int(np.round(abs(maxx-minx)/(self.crop_len*(1-self.overlap))))
        n_steps_y = int(np.round(abs(maxy-miny)/(self.crop_len*(1-self.overlap))))

        # calculate the steps. Stagger the array based on self.crop_len/2 to get the 
        # center of the cell
        x_samp = np.linspace(start=minx,
                           stop=maxx,
                           num=n_steps_x, endpoint=False)+self.crop_len/2

        y_samp = np.linspace(start=miny,
                           stop=maxy,
                           num=n_steps_y, endpoint=False)+self.crop_len/2


        # clip rasters:
        for east in x_samp:
            # TODO: something better than looping?
            for north in y_samp:
                # dictionary to save sample information
                sample_dict = {} 

                # TODO: something better than looping?
                for source_name, source in self.dict_xar[idx].items():
                    
                    # use masks
                    mask_x = (source.x >= east-self.crop_len/2) & (source.x <= east+self.crop_len/2)
                    mask_y = (source.y >= north-self.crop_len/2) & (source.y <= north+self.crop_len/2)

                    # save crop
                    sample_dict[source_name] = source[dict(x=mask_x, y=mask_y)]

                # save sample information in the full sample_dict
                self.samples_dict_full[len(self.samples_dict_full)] = sample_dict.copy()
        
        return self.samples_dict_full.copy()


    def write_full(self):
        """
        TODO: fix
        write the full IO data
        """
        pass
