type MouseEventDetails = { x: number; y: number };
type ClickEventDetails = MouseEventDetails & { button: number };
type KeyEventDetails = { key: string };
type MessageEventDetails = { message: string };

type EventType = keyof DocumentEventMap
  | 'f12_pressed'
  | 'escape_pressed'
  | 'enter_key_press'
  | 'replay_trigger';

type EventDetails =
  | MouseEventDetails
  | ClickEventDetails
  | KeyEventDetails
  | MessageEventDetails;

interface CustomEventLog {
  type: EventType;
  details: EventDetails;
  timestamp: number;
}
