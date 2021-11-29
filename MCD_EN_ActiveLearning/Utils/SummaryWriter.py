from tensorboardX import SummaryWriter

class LogSummary():

    def __init__(self,name):
        self.writer = SummaryWriter('logs/' + name, flush_secs=1)

    def write_final_accuracy(self, accuracy, round):
        self.writer.add_scalar('Round-TestAccuracy', accuracy, round)

    def write_final_ece(self, ece, round):
        self.writer.add_scalar('Round-ECE', ece, round)

    def write_final_r2(self, r2, round):
        self.writer.add_scalar('Round-R2', r2, round)

    def write_per_round_accuracy(self, accuracy, round, epoch):
        self.writer.add_scalar('Round/'+str(round)+'/Accuracy', accuracy, epoch)

    def per_round_layer_output(self, layer_sz, layer_op, round):
        self.writer.add_histogram('/Layer-'+str(layer_sz), layer_op, round)
