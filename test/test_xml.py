import unittest


from graphtage.utils import Tempfile
from graphtage.xml import XML


class TestXML(unittest.TestCase):
    def test_infinite_loop(self):
        """Reproduces https://github.com/trailofbits/graphtage/issues/32"""
        xml = XML.default_instance
        one_xml = b"""
<root>
  <parent>
    <child1 attribute1="foo">child1</child1>
    <child2>child2</child2>
  </parent>
</root>
"""
        two_xml = b"""
<root>
  <parent>
    <child1 attribute1="bar">child1</child1>
    <child2>child2</child2>
  </parent>
</root>
"""
        with Tempfile(one_xml) as one, Tempfile(two_xml) as two:
            t1 = xml.build_tree(one)
            t2 = xml.build_tree(two)
            for edit in t1.get_all_edits(t2):
                print(edit)
