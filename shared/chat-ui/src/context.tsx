import { createContext, useContext } from "react";

import type { SpanMarker } from "./types";

export type ChatUiConfig = {
  spanMarker?: SpanMarker | null;
};

export const ChatUiConfigContext = createContext<ChatUiConfig>({});

export const useChatUiConfig = (): ChatUiConfig => useContext(ChatUiConfigContext);
