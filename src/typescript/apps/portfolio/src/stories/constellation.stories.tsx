import type { Meta, StoryObj } from "@storybook/nextjs-vite";

import { Constellation } from "@/components/constellation";

const meta = {
  title: "Theme/Constellation",
  component: Constellation,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ display: "grid", minHeight: "36rem", padding: "3rem", placeItems: "center" }}>
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof Constellation>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Full: Story = { args: { compact: false } };
export const Compact: Story = { args: { compact: true } };
