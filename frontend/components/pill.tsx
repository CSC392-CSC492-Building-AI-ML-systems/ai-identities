type PillProps = {
  text: string;
};

export default function Pill({ text }: PillProps) {
  return (
    <span className="inline-block bg-white text-black px-8 py-2 rounded-full text-sm select-none">
      {text}
    </span>
  );
}
