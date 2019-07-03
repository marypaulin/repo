class Server:
    def __init__(self, server_id, queue, table):
        while table.get('__terminate__') == None:
            queue.service()
            table.service()