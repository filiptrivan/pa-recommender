from io import StringIO

# https://stackoverflow.com/questions/2414667/python-string-class-like-stringbuilder-in-c
class StringBuilder:
     _file_str = None

     def __init__(self):
         self._file_str = StringIO()

     def append(self, s):
         self._file_str.write(s)

     def __str__(self):
         return self._file_str.getvalue()