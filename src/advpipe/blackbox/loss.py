
def margin_loss(loss_val, margin):
    return max(loss_val + margin, 0)


LOSS_FUNCTIONS = {"margin_loss":margin_loss}