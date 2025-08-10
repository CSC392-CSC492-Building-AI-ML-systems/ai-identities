export default function WikiPage() {
  return (
    <main className="min-h-screen bg-[#050a1f] pt-20 flex flex-col items-center">
      <div className="w-full flex-1 flex justify-center">
        <iframe
          src="http://159.203.20.200:8080/bin/view/Main"
          title="LLMDetective Wiki"
          className="w-full max-w-7xl h-[80vh] rounded-xl border-2 border-[#2D2A5A] bg-white"
          style={{ minHeight: 400 }}
        />
      </div>
    </main>
  );
}
