import copy


class BinInnerParam(object):
    """
    Use to store columns related params for binning process
    """

    def __init__(self):
        self.bin_indexes = []
        self.bin_names = []
        self.col_name_maps = {}
        self.header = []
        self.transform_bin_indexes = []
        self.transform_bin_names = []
        self.category_indexes = []
        self.category_names = []

    def set_header(self, header):
        self.header = copy.deepcopy(header)
        for idx, col_name in enumerate(self.header):
            self.col_name_maps[col_name] = idx

    def set_bin_all(self):
        """
        Called when user set to bin all columns
        """
        self.bin_indexes = [i for i in range(len(self.header))]
        self.bin_names = copy.deepcopy(self.header)

    def set_transform_all(self):
        self.transform_bin_indexes = self.bin_indexes
        self.transform_bin_names = self.bin_names
        self.transform_bin_indexes.extend(self.category_indexes)
        self.transform_bin_names.extend(self.category_names)

    def add_bin_indexes(self, bin_indexes):
        if bin_indexes is None:
            return
        for idx in bin_indexes:
            if idx >= len(self.header):
                # LOGGER.warning("Adding a index that out of header's bound")
                # continue
                raise ValueError("Adding a index that out of header's bound")
            if idx not in self.bin_indexes:
                self.bin_indexes.append(idx)
                self.bin_names.append(self.header[idx])

    def add_bin_names(self, bin_names):
        if bin_names is None:
            return

        for bin_name in bin_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.bin_indexes:
                self.bin_indexes.append(idx)
                self.bin_names.append(self.header[idx])

    def add_transform_bin_indexes(self, transform_indexes):
        if transform_indexes is None:
            return

        for idx in transform_indexes:
            if idx >= len(self.header) or idx < 0:
                raise ValueError("Adding a index that out of header's bound")
                # LOGGER.warning("Adding a index that out of header's bound")
                # continue
            if idx not in self.transform_bin_indexes:
                self.transform_bin_indexes.append(idx)
                self.transform_bin_names.append(self.header[idx])

    def add_transform_bin_names(self, transform_names):
        if transform_names is None:
            return
        for bin_name in transform_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                raise ValueError("Adding a col_name that is not exist in header")

            if idx not in self.transform_bin_indexes:
                self.transform_bin_indexes.append(idx)
                self.transform_bin_names.append(self.header[idx])

    def add_category_indexes(self, category_indexes):
        if category_indexes == -1:
            category_indexes = [i for i in range(len(self.header))]
        elif category_indexes is None:
            return

        for idx in category_indexes:
            if idx >= len(self.header):
                # LOGGER.warning("Adding a index that out of header's bound")
                continue
            if idx not in self.category_indexes:
                self.category_indexes.append(idx)
                self.category_names.append(self.header[idx])
            if idx in self.bin_indexes:
                self.bin_indexes.remove(idx)
                self.bin_names.remove(self.header[idx])

    def add_category_names(self, category_names):
        if category_names is None:
            return

        for bin_name in category_names:
            idx = self.col_name_maps.get(bin_name)
            if idx is None:
                # LOGGER.warning("Adding a col_name that is not exist in header")
                continue
            if idx not in self.category_indexes:
                self.category_indexes.append(idx)
                self.category_names.append(self.header[idx])
            if idx in self.bin_indexes:
                self.bin_indexes.remove(idx)
                self.bin_names.remove(self.header[idx])

    @property
    def bin_cols_map(self):
        assert len(self.bin_indexes) == len(self.bin_names)
        return dict(zip(self.bin_names, self.bin_indexes))

    def encode_col_name_dict(self, col_name_dict: dict, model):
        result = {}
        for x, y in col_name_dict.items():
            col_index = self.col_name_maps.get(x)
            result[anonymous_generator.generate_anonymous(col_index, model=model)] = y
        return result

    def encode_col_name_list(self, col_name_list: list, model):
        result = []
        for x in col_name_list:
            col_index = self.col_name_maps.get(x)
            result.append(anonymous_generator.generate_anonymous(col_index, model=model))
        return result

    # def __encode_col_name(self, col_name):
    #     col_index = self.col_name_maps.get(col_name)
    #     if col_index is None:
    #         LOGGER.warning("Encoding a non-exist column name")
    #         return None
    #     return '.'.join(['host', str(col_index)])

    def decode_col_name(self, encoded_name: str):
        col_index = anonymous_generator.reconstruct_fid(encoded_name)

        # try:
        #     col_index = int(encoded_name.split('.')[1])
        # except IndexError or ValueError:
        #     raise RuntimeError("Bin inner param is trying to decode an invalid col_name.")
        return self.header[col_index]