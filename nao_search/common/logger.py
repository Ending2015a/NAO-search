import os
import sys
import time
import logging

from collections import OrderedDict

class Logger:
    def __init__(self):
        self.LOG = logging.getLogger()

        self.header = None
        self.groups = OrderedDict()
        self.groups['None'] = OrderedDict()

        self.current_group = 'None'

    def set_header(self, name=None):
        if not isinstance(name, str):
            raise ValueError("Header name must be 'str' type")
        self.header = name

    def switch_group(self, name=None):

        if name is None:
            name = 'None'

        if not isinstance(name, str):
            raise ValueError("Group name must be 'str' type")

        self.current_group = name

    def add_pair(self, key, value):

        if not self.current_group in self.groups:
            self.groups[self.current_group] = OrderedDict()

        self.groups[self.current_group][key] = value


    def _create_header(self, width):
        if self.header is None:
            header = '=' * width

        else:
            half = (width - len(self.header)) // 2 - 1
            header = ' '.join(['=' * half, self.header, '=' * half])

        return header

    def _create_tail(self, width):
        return '=' * width

    def _create_subheader(self, name, width):

        #'======= Epoch 5/10 ======='
        #'|------- Training -------|'
        #'| loss: 100.0            |'
        #'| entropy: 1132121.456   |'
        #'|--------- Eval ---------|'
        #'| loss: 2121.17455       |'
        #'=========================='

        remain = width - len(name)
        l_half = remain // 2
        r_half = remain - l_half

        l_half -= 2
        r_half -= 2

        subheader = ''.join(['|', '-'*l_half, ' ', name, ' ', '-'*r_half, '|'])

        return subheader

    def _create_row(self, string, width):
        remain = width - len(string) - 4

        return ''.join(['| ', string, ' '*remain, ' |'])

    def dump_to_log(self):

        kv_strs = OrderedDict()

        max_length = 0

        for group, pairs in self.groups.items():
            
            if group not in kv_strs:
                kv_strs[group] = []

            for key, value in pairs.items():
                string = '{}: {}'.format(key, value)

                kv_strs[group].append(string)
                max_length = max_length if len(string) < max_length else len(string)

        max_width = max_length + len('|  |')


        for group in self.groups.keys():
            group_width = len(group) + len('|-  -|')

            max_width = max_width if group_width < max_width else group_width


        if self.header is not None:
            header_width = len(self.header) + len('=====  =====')

            max_width = max_width if header_width < max_width else header_width

            if (max_width - len(self.header)) % 2 > 0:
                max_width += 1


        self.LOG.info(self._create_header(max_width))

        for group, strings in kv_strs.items():
            
            if len(strings) == 0:
                continue

            if group is not 'None':
                self.LOG.info(self._create_subheader(group, max_width))

            for contant in strings:
                self.LOG.info(self._create_row(contant, max_width))        

        self.groups = OrderedDict()
        self.groups['None'] = OrderedDict()
