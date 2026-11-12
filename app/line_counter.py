
class LineCounter:

    def __init__(
        self,
        upper_position=0.35,
        lower_position=0.65
    ):

        self.upper_position = upper_position
        self.lower_position = lower_position

        self.entries = 0
        self.exits = 0

        self.track_states = {}

    def update(self, tracked, frame_height):

        upper_line = int(
            frame_height * self.upper_position
        )

        lower_line = int(
            frame_height * self.lower_position
        )

        if tracked.tracker_id is None:
            return

        for i in range(len(tracked)):

            track_id = int(
                tracked.tracker_id[i]
            )

            if track_id < 0:
                continue

            x1, y1, x2, y2 = tracked.xyxy[i]

            center_y = int(
                (y1 + y2) / 2
            )

            # Identifica a posição da pessoa
            if center_y < upper_line:

                current_side = -1

            elif center_y > lower_line:

                current_side = 1

            else:

                current_side = 0

            # Primeiro contato com a pessoa
            if track_id not in self.track_states:

                self.track_states[track_id] = {
                    "side": current_side,
                    "last_y": center_y
                }

                continue

            state = self.track_states[track_id]

            previous_side = state["side"]
            previous_y = state["last_y"]

            direction = None

            # Travessia completa da zona
            if (
                previous_side == -1
                and current_side == 1
            ):

                direction = "IN"

            elif (
                previous_side == 1
                and current_side == -1
            ):

                direction = "OUT"

            # Pessoa entrou na zona e saiu por um lado
            elif (
                previous_side == 0
                and current_side == 1
            ):

                if center_y > previous_y:
                    direction = "IN"

            elif (
                previous_side == 0
                and current_side == -1
            ):

                if center_y < previous_y:
                    direction = "OUT"

            # Registra a travessia
            if direction == "IN":

                self.entries += 1

                print(
                    f"IN | Pessoa #{track_id}"
                )

            elif direction == "OUT":

                self.exits += 1

                print(
                    f"OUT | Pessoa #{track_id}"
                )

            # Atualiza o lado somente fora da zona
            if current_side != 0:

                state["side"] = current_side

            state["last_y"] = center_y

    @property
    def people_inside(self):

        return self.entries - self.exits
