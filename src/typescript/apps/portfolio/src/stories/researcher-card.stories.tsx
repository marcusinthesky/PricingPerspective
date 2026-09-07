import type { Meta, StoryObj } from "@storybook/nextjs-vite";

import { ResearcherCard } from "@/components/researcher-card";
import { researchers } from "@/content/data";

const meta = {
  title: "Research/Researcher card",
  component: ResearcherCard,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ maxWidth: "48rem", padding: "3rem" }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof ResearcherCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Marcus: Story = { args: { researcher: researchers[0] } };
export const ChunSung: Story = { args: { researcher: researchers[1] } };
