import asyncio


class Worker:
    def __init__(self, scheduled_event: asyncio.Event, result_publish:  asyncio.Event):
        self.scheduled_event = scheduled_event
        self.result_publish = result_publish

    async def run(self) -> None:
        for it in range(1, 4):
            print("Worker iter:", it)
            worker_co = self.worker_subroutine(it)
            print(f"Worker coroutine: {worker_co}, type = {type(worker_co)}")
            ret = await worker_co

            print(f"subroutine finished, sum of [1, ..., {it}] = {ret}")

    async def worker_subroutine(self, n: int) -> int:
        partial_sum = 0
        for j in range(1, n+1):
            # wait to be scheduled
            print("worker_subroutine - waiting to be scheduled")

            raise Exception("This is my pee")

            print("self.scheduled_event.is_set(): ", self.scheduled_event.is_set())
            print(id(self.scheduled_event))

            await self.scheduled_event.wait()
            self.scheduled_event.clear()
            print("worker scheduled!")
            partial_sum += j
            print(f"j: {j}, partial sum: {partial_sum}")

            assert not self.result_publish.is_set()
            # Publish result
            self.result_publish.set()


        return partial_sum
            

async def main() -> None:
    # Create an Event object.
    scheduled_event = asyncio.Event()
    result_publish = asyncio.Event()
    worker = Worker(scheduled_event, result_publish)

    # Spawn a Task to wait until 'event' is set.
    worker_task = asyncio.create_task(worker.run())

    while not worker_task.done():
        # do 1 step
        print("Manager is scheduling worker")
        scheduled_event.set()

        print(id(scheduled_event), id(worker.scheduled_event))

        print("worker.scheduled_event.is_set:", worker.scheduled_event.is_set())

        await result_publish.wait()
        print("Manager got result")
        result_publish.clear()

    await worker_task

asyncio.run(main())