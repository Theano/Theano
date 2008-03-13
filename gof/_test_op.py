
import unittest
from copy import copy
from op import *
from result import ResultBase #, BrokenLinkError


class MyResult(ResultBase):

    def __init__(self, thingy):
        self.thingy = thingy
        ResultBase.__init__(self, role = None)
        self.data = [self.thingy]

    def __eq__(self, other):
        return self.same_properties(other)

    def same_properties(self, other):
        return isinstance(other, MyResult) and other.thingy == self.thingy

    def __str__(self):
        return str(self.thingy)

    def __repr__(self):
        return str(self.thingy)


class MyOp(Op):

    def __init__(self, *inputs):
        for input in inputs:
            if not isinstance(input, MyResult):
                raise Exception("Error 1")
        self.inputs = inputs
        self.outputs = [MyResult(sum([input.thingy for input in inputs]))]

#     def validate_update(self):
#         for input in self.inputs:
#             if not isinstance(input, MyResult):
#                 raise Exception("Error 1")
#         if self.outputs is None:
#             self.outputs = [MyResult(sum([input.thingy for input in self.inputs]))]
#             return True
#         else:
#             old_thingy = self.outputs[0].thingy
#             new_thingy = sum([input.thingy for input in self.inputs])
#             self.outputs[0].thingy = new_thingy
#             return old_thingy != new_thingy

# class MyOp(Op):

#     def validate_update(self):
#         for input in self.inputs:
#             if not isinstance(input, MyResult):
#                 raise Exception("Error 1")
#         self.outputs = [MyResult(sum([input.thingy for input in self.inputs]))]


class _test_Op(unittest.TestCase):

    # Sanity tests
    def test_sanity_0(self):
        r1, r2 = MyResult(1), MyResult(2)
        op = MyOp(r1, r2)
        assert op.inputs == [r1, r2] # Are the inputs what I provided?
        assert op.outputs == [MyResult(3)] # Are the outputs what I expect?
        assert op.outputs[0].owner is op and op.outputs[0].index == 0

    # validate_update
    def test_validate_update(self):
        try:
            MyOp(ResultBase(), MyResult(1)) # MyOp requires MyResult instances
        except Exception, e:
            assert str(e) == "Error 1"
        else:
            raise Exception("Expected an exception")

#     # Setting inputs and outputs
#     def test_set_inputs(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         r3 = op.outputs[0]
#         op.inputs = MyResult(4), MyResult(5)
#         op.validate_update()
#         assert op.outputs == [MyResult(9)] # check if the output changed to what I expect
# #         assert r3.data is op.outputs[0].data # check if the data was properly transferred by set_output
        
#     def test_set_bad_inputs(self):
#         op = MyOp(MyResult(1), MyResult(2))
#         try:
#             op.inputs = MyResult(4), ResultBase()
#             op.validate_update()
#         except Exception, e:
#             assert str(e) == "Error 1"
#         else:
#             raise Exception("Expected an exception")

#     def test_set_outputs(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2) # here we only make one output
#         try:
#             op.outputs = MyResult(10), MyResult(11) # setting two outputs should fail
#         except TypeError, e:
#             assert str(e) == "The new outputs must be exactly as many as the previous outputs."
#         else:
#             raise Exception("Expected an exception")
        

#     # Tests about broken links
#     def test_create_broken_link(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         r3 = op.out
#         op.inputs = MyResult(3), MyResult(4)
#         assert r3 not in op.outputs
#         assert r3.replaced
        
#     def test_cannot_copy_when_input_is_broken_link(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         r3 = op.out
#         op2 = MyOp(r3)
#         op.inputs = MyResult(3), MyResult(4)
#         try:
#             copy(op2)
#         except BrokenLinkError:
#             pass
#         else:
#             raise Exception("Expected an exception")
        
#     def test_get_input_broken_link(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         r3 = op.out
#         op2 = MyOp(r3)
#         op.inputs = MyResult(3), MyResult(4)
#         try:
#             op2.get_input(0)
#         except BrokenLinkError:
#             pass
#         else:
#             raise Exception("Expected an exception")
        
#     def test_get_inputs_broken_link(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         r3 = op.out
#         op2 = MyOp(r3)
#         op.inputs = MyResult(3), MyResult(4)
#         try:
#             op2.get_inputs()
#         except BrokenLinkError:
#             pass
#         else:
#             raise Exception("Expected an exception")

#     def test_repair_broken_link(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         r3 = op.out
#         op2 = MyOp(r3, MyResult(10))
#         op.inputs = MyResult(3), MyResult(4)
#         op2.repair()
#         assert op2.outputs == [MyResult(17)]

#     # Tests about string representation
#     def test_create_broken_link(self):
#         r1, r2 = MyResult(1), MyResult(2)
#         op = MyOp(r1, r2)
#         assert str(op) == "MyOp(1, 2)"



if __name__ == '__main__':
    unittest.main()
