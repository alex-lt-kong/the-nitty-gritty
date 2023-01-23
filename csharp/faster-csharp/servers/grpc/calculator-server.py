from concurrent import futures
import logging

import grpc
import calculator_pb2
import calculator_pb2_grpc


class Calculator(calculator_pb2_grpc.CalculatorServicer):

    def add(self, request, context):
        return calculator_pb2.CalculatorReply(result=request.num1 + request.num2)
    def minus(self, request, context):
        return calculator_pb2.CalculatorReply(result=request.num1 - request.num2)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calculator_pb2_grpc.add_CalculatorServicer_to_server(Calculator(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
