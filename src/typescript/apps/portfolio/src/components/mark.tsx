import Image from "next/image";

import { sitePath } from "@/lib/site";

export function Mark() {
  return (
    <Image
      className="mark"
      src={sitePath("/icons/atlas-mark.svg")}
      alt=""
      width={42}
      height={42}
      unoptimized
    />
  );
}
